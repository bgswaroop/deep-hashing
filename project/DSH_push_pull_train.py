import argparse
from pathlib import Path
from typing import Any, Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets.cifar import CIFAR10

from utils.loss_function import dsh_loss
from utils.metrics import compute_map_score


class PushPullBase(pl.LightningModule):
    def __init__(self):
        super(PushPullBase, self).__init__()

    def training_step(self, batch, batch_idx):
        x, y = batch
        b = self(x)
        loss = dsh_loss(b, y, self.hparams.hash_length * 2, self.hparams.regularization_weight_alpha)
        self.log('loss', {'train': loss})
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        x, y = batch
        predictions = self(x)
        hash_code = torch.sign(predictions)

        if dataloader_idx == 0:
            loss = dsh_loss(predictions, y, self.hparams.hash_length * 2, self.hparams.regularization_weight_alpha)
            self.log('loss', {'val': loss}, add_dataloader_idx=False)
            # This additional log is to save the model with the least loss_val
            self.log('loss_val', loss, add_dataloader_idx=False, logger=False)
            return {'val_data': {'hash_codes': hash_code, 'ground_truths': y}}
        elif dataloader_idx == 1:
            return {'train_data': {'hash_codes': hash_code, 'ground_truths': y}}
        elif dataloader_idx == 2:
            return {'test_data': {'hash_codes': hash_code, 'ground_truths': y}}

    def validation_epoch_end(self, outputs) -> None:
        val_hash_codes = torch.concat([x['val_data']['hash_codes'] for x in outputs[0]])
        val_ground_truths = torch.concat([x['val_data']['ground_truths'] for x in outputs[0]])
        trn_hash_codes = torch.concat([x['train_data']['hash_codes'] for x in outputs[1]])
        trn_ground_truths = torch.concat([x['train_data']['ground_truths'] for x in outputs[1]])
        test_hash_codes = torch.concat([x['test_data']['hash_codes'] for x in outputs[2]])
        test_ground_truths = torch.concat([x['test_data']['ground_truths'] for x in outputs[2]])

        val_score = compute_map_score(trn_hash_codes, trn_ground_truths, val_hash_codes, val_ground_truths, self.device)
        # train_score = compute_map_score(trn_hash_codes, trn_ground_truths, trn_hash_codes, trn_ground_truths,
        # self.device)
        test_score = compute_map_score(trn_hash_codes, trn_ground_truths, test_hash_codes, test_ground_truths,
                                       self.device)

        self.log('MAP_score', {'val': val_score, 'test': test_score})

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        x, y = batch
        predictions = self(x)
        hash_code = torch.sign(predictions)
        return {'hash_codes': hash_code, 'ground_truths': y}

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.hparams.learning_rate,
                                     weight_decay=self.hparams.weight_decay)

        # decrease learning rate by 40% (gamma) after 20,000 iterations or 40 epochs (step_size)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=1 - 0.4)

        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--hash_length', type=int, default=48)
        parser.add_argument('--learning_rate', type=float, default=1e-3)
        parser.add_argument('--weight_decay', type=float, default=0.004)
        parser.add_argument('--regularization_weight_alpha', type=float, default=0.01)
        return parser


class PushPullConv2DUnit(torch.nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            # kernel_size: _size_2_t,
            push_kernel_size: _size_2_t,
            pull_kernel_size: _size_2_t,
            avg_kernel_size: _size_2_t,
            pull_inhibition_strength: int = 1,
            scale_the_outputs: bool = False,
            stride: _size_2_t = 1,
            padding: Union[str, _size_2_t] = 0,
            dilation: _size_2_t = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',
            device=None,
            dtype=None):
        super(PushPullConv2DUnit, self).__init__()

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.pull_inhibition_strength = pull_inhibition_strength
        self.scale_the_outputs = scale_the_outputs
        self.pull_kernel_size = pull_kernel_size

        self.push_conv = torch.nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=push_kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=False, padding_mode=padding_mode, device=device,
            dtype=dtype)

        if avg_kernel_size != 0:
            self.avg = torch.nn.AvgPool2d(
                kernel_size=avg_kernel_size,
                stride=1,
                padding=tuple([int((x - 1) / 2) for x in _pair(avg_kernel_size)])
            )
        else:
            self.avg = None

        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_channels, device=device, dtype=dtype))
            self.bias.data.uniform_(-1, 1)  # random weight initialization
        else:
            self.bias = None

    @property
    def weight(self):
        return self.push_conv.weight

    @weight.setter
    def weight(self, value):
        self.push_conv.weight = value

    def forward(self, x):
        push_response = F.relu_(self.push_conv(x))
        pull_conv_kernel = F.interpolate(-self.push_conv.weight, size=self.pull_kernel_size, mode='bilinear')
        pull_response = F.relu_(F.conv2d(input=x, weight=pull_conv_kernel, stride=self.stride, padding=self.padding,
                                         dilation=self.dilation, groups=self.groups))

        if self.avg:
            avg_pull_response = self.avg(pull_response)
            x_out = F.relu_(push_response - self.pull_inhibition_strength * avg_pull_response)
        else:
            avg_pull_response = None
            x_out = F.relu_(push_response - self.pull_inhibition_strength * pull_response)

        if self.scale_the_outputs:
            ratio = torch.amax(push_response, dim=(2, 3), keepdim=True) / \
                    (torch.amax(x_out, dim=(2, 3), keepdim=True) + 1e-20)
            x_out_scaled = x_out * ratio
        else:
            x_out_scaled = x_out

        if self.bias is not None:
            x_out_scaled = x_out_scaled + self.bias.view((1, -1, 1, 1))

        # plot_push_pull_kernels(push_response, pull_response, avg_pull_response, x, x_out, x_out_scaled, k=0)
        return x_out_scaled


class BaselineConvNet(PushPullBase):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        if args.use_push_pull and args.num_push_pull_layers >= 1:
            self.conv1 = PushPullConv2DUnit(in_channels=3, out_channels=32,
                                            push_kernel_size=args.push_kernel_size,
                                            pull_kernel_size=args.pull_kernel_size,
                                            avg_kernel_size=args.avg_kernel_size,
                                            pull_inhibition_strength=args.pull_inhibition_strength,
                                            scale_the_outputs=args.scale_the_outputs,
                                            padding='same', bias=args.bias)
        else:
            self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5, 5), padding='same')

        if args.use_push_pull and args.num_push_pull_layers >= 2:
            self.conv2 = PushPullConv2DUnit(in_channels=32, out_channels=32,
                                            push_kernel_size=args.push_kernel_size,
                                            pull_kernel_size=args.pull_kernel_size,
                                            avg_kernel_size=args.avg_kernel_size,
                                            pull_inhibition_strength=args.pull_inhibition_strength,
                                            scale_the_outputs=args.scale_the_outputs,
                                            padding='same', bias=args.bias)
        else:
            self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5), padding='same')

        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), padding='same')
        self.fc1 = torch.nn.Linear(in_features=3 * 3 * 64, out_features=500)
        self.fc2 = torch.nn.Linear(in_features=500, out_features=self.hparams.hash_length)

        for layer in [self.conv1, self.conv2, self.conv3, self.fc1, self.fc2]:
            torch.nn.init.xavier_uniform_(layer.weight)

        self.maxPool = torch.nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))

    def forward(self, x):
        x = torch.relu(self.maxPool(self.conv1(x)))
        x = torch.relu(self.maxPool(self.conv2(x)))
        x = torch.relu(self.maxPool(self.conv3(x)))
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class AlexNet(PushPullBase):
    def __init__(self):
        super(AlexNet, self).__init__()

    def forward(self, *args, **kwargs) -> Any:
        raise NotImplementedError


def train_on_clean_images():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=100, type=int)
    # effective batch size is 5050 - as we are performing online training - combinations_with_replacement(batch, 2)
    parser.add_argument('--dataset_dir', default='/data/p288722/datasets/cifar', type=str)
    parser.add_argument('--finetune', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--finetune_ckpt', type=str, default=None)
    parser.add_argument('--ckpt', type=str, default=None)

    # Pytorch lightning args
    parser.add_argument('--logs_dir', required=True, type=str, help='Path to save the logs/metrics during training')
    parser.add_argument('--experiment_name', default='dsh_push_pull', type=str)
    parser.add_argument('--num_workers', default=2, type=int,
                        help='how many subprocesses to use for data loading. ``0`` means that the data will be '
                             'loaded in the main process. (default: ``2``)')
    parser.add_argument('--logs_version', default=None, type=int)

    # Push Pull Convolutional Unit Params
    parser.add_argument('--use_push_pull', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--num_push_pull_layers', type=int, default=1)
    parser.add_argument('--push_kernel_size', type=int, default=3, help='Size of the push filter (int)')
    parser.add_argument('--pull_kernel_size', type=int, default=3, help='Size of the pull filter (int)')
    parser.add_argument('--avg_kernel_size', type=int, default=3, help='Size of the avg filter (int)')
    parser.add_argument('--pull_inhibition_strength', type=float, default=1.0)
    parser.add_argument('--scale_the_outputs', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--bias', default=True, action=argparse.BooleanOptionalAction)

    parser = pl.Trainer.add_argparse_args(parser)
    parser = BaselineConvNet.add_model_specific_args(parser)
    args = parser.parse_args()

    if not args.accelerator:
        args.accelerator = 'gpu'
    if args.accelerator == 'gpu':
        args.device = torch.device(f'cuda:{torch.cuda.current_device()}')
    else:
        args.device = torch.device(f'cpu')
    if not args.max_epochs:
        args.max_epochs = 60

    assert Path(args.dataset_dir).exists(), 'dataset_dir does not exists!'
    assert Path(args.logs_dir).exists(), 'logs_dir does not exists!'
    Path(args.logs_dir).joinpath(args.experiment_name).mkdir(exist_ok=True)

    if args.finetune:
        assert Path(args.finetune_ckpt).exists(), 'finetune_ckpt path does not exists!'

    # ------------
    # data
    # ------------
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    dataset = CIFAR10(args.dataset_dir, train=True, download=True, transform=transform_train)
    data_split_test = CIFAR10(args.dataset_dir, train=False, download=True, transform=transform_test)
    data_split_train, data_split_val = random_split(dataset, [45000, 5000], generator=torch.Generator().manual_seed(99))
    train_loader = DataLoader(data_split_train, batch_size=args.batch_size, num_workers=args.num_workers)
    val_loader = DataLoader(data_split_val, batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader = DataLoader(data_split_test, batch_size=args.batch_size, num_workers=args.num_workers)

    # ------------
    # model
    # ------------
    model = BaselineConvNet(args)

    if args.finetune:
        # copy the pre-trained weights from 12-bit trained model to the 48-bit model
        model_12_bit = torch.load(args.finetune_ckpt)
        layers_to_update = set(list(model_12_bit['state_dict'].keys())[:-2])
        for name, param in model.state_dict().items():
            if name in layers_to_update:
                param.copy_(model_12_bit['state_dict'][name])

    # ------------
    # training
    # ------------
    logger = TensorBoardLogger(save_dir=args.logs_dir, name=args.experiment_name, default_hp_metric=False,
                               version=args.logs_version)
    ckpt_callback = ModelCheckpoint(mode='min', monitor='loss_val',
                                    save_last=True)  # fixme: populate dir_path when continue training is set
    lr_monitor_callback = LearningRateMonitor(logging_interval='epoch')
    trainer = pl.Trainer.from_argparse_args(args, logger=logger, callbacks=[ckpt_callback, lr_monitor_callback],
                                            resume_from_checkpoint=args.ckpt)
    trainer.fit(model, train_loader, val_dataloaders=[val_loader, train_loader, test_loader])

    # ------------
    # testing
    # ------------
    # Optionally add ckpt_path to trainer.predict()
    train_predictions = trainer.predict(model=model, dataloaders=train_loader)
    test_predictions = trainer.predict(model=model, dataloaders=test_loader)

    train_hash_codes = torch.concat([x['hash_codes'] for x in train_predictions])
    train_ground_truths = torch.concat([x['ground_truths'] for x in train_predictions])
    test_hash_codes = torch.concat([x['hash_codes'] for x in test_predictions])
    test_ground_truths = torch.concat([x['ground_truths'] for x in test_predictions])

    map_score = compute_map_score(train_hash_codes, train_ground_truths, test_hash_codes, test_ground_truths,
                                  args.device)
    print(f'MAP score: {map_score}')


if __name__ == '__main__':
    train_on_clean_images()
