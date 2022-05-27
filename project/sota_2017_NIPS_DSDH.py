import argparse
from pathlib import Path
from typing import Any, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as f
import torchvision.models as models
from PIL import Image
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets.cifar import CIFAR10

from .utils.metrics import compute_map_score


class Classifier(pl.LightningModule):
    def __init__(self, *args):
        super(Classifier, self).__init__()
        self.save_hyperparameters(*args, logger=True)

        self._H = None
        self._B = None
        self._Y = None
        self._S = None

    def dsdh_hashing_loss(self, H, S, B) -> torch.tensor:
        """
        Deep Supervised Discrete Hashing (DSDH) Loss
        https://proceedings.neurips.cc/paper/2017/file/e94f63f579e05cb49c05c2d050ead9c0-Paper.pdf
        :param H: img_hash for the current mini batch
        :param S: nearest_neighbours between the current mini batch and the entire training dataset (similarity matrix)
        :param B: binary_img_hash for the current mini batch
        :return: loss
        """
        # metric loss
        psi = torch.clamp(0.5 * H @ self._H.T, min=-100, max=50)
        l1 = - torch.mean(S * psi - torch.log(1 + torch.exp(psi)))
        # quantization loss (with indirect classification loss)
        l4 = torch.mean((B - H) ** 2)
        loss = l1 + self.hparams.eta * l4
        return loss

    def dsdh_full_loss(self, W) -> torch.tensor:
        """
        Deep Supervised Discrete Hashing (DSDH) Loss
        https://proceedings.neurips.cc/paper/2017/file/e94f63f579e05cb49c05c2d050ead9c0-Paper.pdf
        :param W: linear classifier (binary hash to classification targets)
        :return: loss
        """
        H, S, W, Y, B = self._H.to('cpu'), self._S.to('cpu'), W.to('cpu'), self._Y.to('cpu'), self._B.to('cpu')
        # metric loss
        psi = torch.clamp(0.5 * H @ H.T, min=-100, max=50)
        l1 = - torch.mean(S * psi - torch.log(1 + torch.exp(psi)))
        # regularization loss
        l2 = torch.sum(W ** 2)
        # classification loss
        l3 = torch.mean((Y - B @ W) ** 2)
        # quantization loss
        l4 = torch.mean((B - H) ** 2)
        loss = l1 + self.hparams.nu * l2 + self.hparams.mu * l3 + self.hparams.eta * l4
        return loss

    def on_train_start(self) -> None:
        if self._H is None:
            N, K, C = self.hparams.num_train_samples, self.hparams.hash_length, self.hparams.num_categories
            self._H = torch.zeros((N, K), device=self.device, requires_grad=False)
            self._B = torch.randn((N, K), device=self.device, requires_grad=False).sign()
            target_class = self.trainer.train_dataloader.dataset.datasets.targets
            self._Y = f.one_hot(target_class, C).float().to(self.device)
            self._S = (self._Y @ self._Y.T > 0).float().to(self.device)

    def training_step(self, batch, batch_idx):
        X, Y, idx = batch
        H = self(X)
        self._H[idx] = H.clone().detach().requires_grad_(False)
        loss = self.dsdh_hashing_loss(H, self._S[idx], self._B[idx])

        self.log('hashing-loss', {'train': loss})

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        X, Y, idx = batch
        H = self(X)
        hash_code = torch.sign(H)
        Y = torch.argmax(Y, dim=1)

        if dataloader_idx == 0:
            return {'val_data': {'hash_codes': hash_code, 'ground_truths': Y}}
        elif dataloader_idx == 1:
            return {'train_data': {'hash_codes': hash_code, 'ground_truths': Y}}
        elif dataloader_idx == 2:
            return {'test_data': {'hash_codes': hash_code, 'ground_truths': Y}}

    def validation_epoch_end(self, outputs) -> None:
        val_hash_codes = torch.concat([x['val_data']['hash_codes'] for x in outputs[0]])
        val_ground_truths = torch.concat([x['val_data']['ground_truths'] for x in outputs[0]])
        trn_hash_codes = torch.concat([x['train_data']['hash_codes'] for x in outputs[1]])
        trn_ground_truths = torch.concat([x['train_data']['ground_truths'] for x in outputs[1]])
        test_hash_codes = torch.concat([x['test_data']['hash_codes'] for x in outputs[2]])
        test_ground_truths = torch.concat([x['test_data']['ground_truths'] for x in outputs[2]])

        val_score = compute_map_score(trn_hash_codes, trn_ground_truths, val_hash_codes, val_ground_truths)
        test_score = compute_map_score(trn_hash_codes, trn_ground_truths, test_hash_codes, test_ground_truths)
        train_score = compute_map_score(trn_hash_codes, trn_ground_truths, trn_hash_codes, trn_ground_truths)
        self.log('MAP_score', {'train': train_score, 'val': val_score, 'test': test_score})

    def on_train_epoch_end(self) -> None:
        B, Y, H = self._B, self._Y, self._H
        nu, mu, eta = self.hparams.nu, self.hparams.mu, self.hparams.eta
        K = self.hparams.hash_length

        # Compute W (based on B)
        W = torch.inverse(B.T @ B + nu / mu * torch.eye(K).to(self.device)) @ (B.T @ Y)

        # Compute B (based on W)
        P = Y @ W.T + eta / mu * H
        B = torch.zeros_like(B)
        # for _ in range(10):
        #     B0 = B
        # iterating bit-wise over self.hparams.hash_length
        for k in range(K):
            B1 = torch.concat([B[:, :k], B[:, k + 1:]], dim=1)
            W1 = torch.concat([W[:k, :], W[k + 1:, :]], dim=0)
            B[:, k] = torch.sign(P[:, k] - B1 @ W1 @ W[k, :])  # (eq 18)
        # if torch.linalg.norm(B - B0) < 1e-6 * torch.linalg.norm(B0):
        #     break

        # Update B
        self._B = B

        loss = self.dsdh_full_loss(W)
        self.log('hashing-and-classification-loss', {'train': loss})

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        X, Y, idx = batch
        predictions = self(X)
        hash_code = torch.sign(predictions)
        Y = torch.argmax(Y, dim=1)
        return {'hash_codes': hash_code, 'ground_truths': Y}

    def configure_optimizers(self):

        optimizer = torch.optim.RMSprop(self.parameters(),
                                        lr=self.hparams.learning_rate,
                                        weight_decay=self.hparams.weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.hparams.max_epochs, 1e-7)

        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--hash_length', type=int, default=48)
        parser.add_argument('--learning_rate', type=float, default=1e-5)
        parser.add_argument('--eta', type=int, default=1e-2)
        parser.add_argument('--mu', type=float, default=0.01)
        parser.add_argument('--nu', type=float, default=1)

        parser.add_argument('--weight_decay', type=float, default=1e-5)
        return parser


class ClassifierVgg16(Classifier):
    def __init__(self, *args):
        super(ClassifierVgg16, self).__init__(*args)
        self.vgg_model = models.vgg16(pretrained=True)
        self.vgg_model.classifier = self.vgg_model.classifier[:-1]
        self.hash_layer = torch.nn.Sequential(
            torch.nn.Linear(in_features=4096, out_features=self.hparams.hash_length),
            torch.nn.Tanh()
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.vgg_model(x)
        x = self.hash_layer(x)
        return x


class ClassifierCnnF(Classifier):
    """This is an implementation of the CNN-F architecture from the paper -
    Return of the Devil in the Details: Delving Deep into Convolutional Nets (BMVC 2014)
    """

    def __init__(self, *args):
        super(ClassifierCnnF, self).__init__(*args)

        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(11, 11), stride=4, padding='valid')
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(5, 5), stride=1, padding='same')
        self.conv3 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding='same')
        self.conv4 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding='same')
        self.conv5 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding='same')

        self.fc1 = torch.nn.Linear(in_features=9216, out_features=4096)
        self.fc2 = torch.nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = torch.nn.Linear(in_features=4096, out_features=self.hparams.hash_length)

        for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.fc1, self.fc2, self.fc3]:
            torch.nn.init.xavier_uniform_(layer.weight)

        # non-trainable layers
        self.maxPool = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.lrn1 = torch.nn.LocalResponseNorm(size=3)
        self.lrn2 = torch.nn.LocalResponseNorm(size=256)
        self.dropout = torch.nn.Dropout()

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = torch.relu(self.maxPool(self.lrn1(self.conv1(x))))
        x = torch.relu(self.maxPool(self.lrn2(self.conv2(x))))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.maxPool(self.conv5(x)))

        x = torch.flatten(x, start_dim=1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


class CustomCIFAR10(CIFAR10):
    def __init__(self, root: str, split: str, *args, **kwargs):
        assert split in {'train', 'val', 'test'}
        if split == 'train' or split == 'val':
            super().__init__(root, train=True, *args, **kwargs)
        else:
            super().__init__(root, train=False, *args, **kwargs)

        shuffle = torch.randperm(len(self.data), generator=torch.Generator().manual_seed(99))
        self.data = self.data[shuffle]
        self.targets = torch.tensor(self.targets)[shuffle]

        num_train = 45_000
        if split == 'train':
            self.data = self.data[:num_train]
            self.targets = self.targets[:num_train]
        elif split == 'val':
            self.data = self.data[num_train:]
            self.targets = self.targets[num_train:]

        class_ids = sorted(self.class_to_idx.values())
        self.num_categories = len(class_ids)

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (img, target_class, target_neighbours, index)
        """

        img = self.data[index]
        target_class = self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            raise Warning('Target transform is not valid')
        target_class = f.one_hot(target_class, self.num_categories).float()

        return img, target_class, index


def main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--classifier', default='vgg16', type=str, choices=['vgg16', 'cnn-f'])
    parser.add_argument('--dataset_dir', default=Path('/data/p288722/datasets/cifar'), type=Path)
    parser.add_argument('--logs_dir', default=Path('/data/p288722/runtime_data/deep_hashing'), type=Path)
    parser.add_argument('--experiment_name', default='sota_dsdh_baseline_cifar10', type=str)
    parser.add_argument('--num_workers', default=2, type=int,
                        help='how many subprocesses to use for data loading. ``0`` means that the data will be '
                             'loaded in the main process. (default: ``2``)')
    parser = pl.Trainer.add_argparse_args(parser)
    parser = Classifier.add_model_specific_args(parser)
    args = parser.parse_args()

    if not args.accelerator:
        args.accelerator = 'gpu'
    if not args.max_epochs:
        args.max_epochs = 150

    assert args.dataset_dir.exists(), 'dataset_dir does not exists!'
    assert args.logs_dir.exists(), 'logs_dir does not exists!'
    args.logs_dir.joinpath(args.experiment_name).mkdir(exist_ok=True)

    # ------------
    # data
    # ------------
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    args.num_categories = 10
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    mnist_train = CustomCIFAR10(args.dataset_dir, split='train', download=True, transform=transform_train)
    mnist_val = CustomCIFAR10(args.dataset_dir, split='val', download=True, transform=transform_train)
    mnist_test = CustomCIFAR10(args.dataset_dir, split='test', download=True, transform=transform_test)
    args.num_train_samples = len(mnist_train)
    train_loader = DataLoader(mnist_train, batch_size=args.batch_size, num_workers=args.num_workers)
    val_loader = DataLoader(mnist_val, batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader = DataLoader(mnist_test, batch_size=args.batch_size, num_workers=args.num_workers)

    # ------------
    # model
    # ------------
    if args.classifier == 'vgg16':
        model = ClassifierVgg16(args)
    elif args.classifier == 'cnn-f':
        model = ClassifierCnnF(args)
    else:
        raise ValueError('Invalid model!')

    # ------------
    # training
    # ------------
    logger = TensorBoardLogger(save_dir=args.logs_dir, name=args.experiment_name, default_hp_metric=False)
    ckpt_callback = ModelCheckpoint(save_last=True)
    lr_monitor_callback = LearningRateMonitor(logging_interval='epoch')
    trainer = pl.Trainer.from_argparse_args(args, logger=logger, callbacks=[ckpt_callback, lr_monitor_callback])
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

    map_score = compute_map_score(train_hash_codes, train_ground_truths, test_hash_codes, test_ground_truths)
    print(f'MAP score: {map_score}')


if __name__ == '__main__':
    main()
