import argparse
import json
from pathlib import Path

import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets.cifar import CIFAR10

from DSH_push_pull_train import Classifier
from utils.metrics import compute_map_score


class AddGaussianNoise(object):
    def __init__(self, mean, std, seed=None):
        self.std = std
        self.mean = mean
        self.seed = seed

    def __call__(self, tensor):
        if self.seed:
            generator = torch.random.manual_seed(0)
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def predict_with_noise():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--dataset_dir', default=Path('/data/p288722/datasets/cifar'), type=Path)
    parser.add_argument('--num_workers', default=2, type=int,
                        help='how many subprocesses to use for data loading. ``0`` means that the data will be '
                             'loaded in the main process. (default: ``2``)')
    parser.add_argument('--model_ckpt', type=str, required=True)
    parser.add_argument('--noise_type', default='gaussian', type=str, choices=['gaussian'])
    parser.add_argument('--use_push_pull', action=argparse.BooleanOptionalAction, default=True)

    parser = pl.Trainer.add_argparse_args(parser)
    parser = Classifier.add_model_specific_args(parser)
    args = parser.parse_args()

    if not args.accelerator:
        args.accelerator = 'gpu'

    assert args.dataset_dir.exists(), 'dataset_dir does not exists!'
    assert Path(args.model_ckpt).exists(), 'model_ckpt path does not exists!'

    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    dataset = CIFAR10(args.dataset_dir, train=True, download=True, transform=transform_train)
    data_split_train, data_split_val = random_split(dataset, [45000, 5000], generator=torch.Generator().manual_seed(99))
    train_loader = DataLoader(data_split_train, batch_size=args.batch_size, num_workers=args.num_workers)

    model_ckpt = torch.load(args.model_ckpt)
    args.hash_length = model_ckpt['hyper_parameters']['hash_length']
    args.push_kernel_size = model_ckpt['hyper_parameters'].get('push_kernel_size', None)
    args.pull_kernel_size = model_ckpt['hyper_parameters'].get('pull_kernel_size', None)
    args.avg_kernel_size = model_ckpt['hyper_parameters'].get('avg_kernel_size', None)
    args.bias = model_ckpt['hyper_parameters'].get('bias', None)
    args.pull_inhibition_strength = model_ckpt['hyper_parameters'].get('pull_inhibition_strength', None)
    args.scale_the_outputs = model_ckpt['hyper_parameters'].get('scale_the_outputs', None)

    model = Classifier(args)
    model.load_state_dict(model_ckpt['state_dict'])

    trainer = pl.Trainer.from_argparse_args(args)
    train_predictions = trainer.predict(model=model, dataloaders=train_loader)
    train_hash_codes = torch.concat([x['hash_codes'] for x in train_predictions])
    train_ground_truths = torch.concat([x['ground_truths'] for x in train_predictions])

    num_tests = 20.0
    mAP_scores = []
    std_devs = [float(x) for x in torch.arange(0, 0.20, 0.20 / num_tests)]
    for std in std_devs:
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            AddGaussianNoise(mean=0, std=std),
            normalize,
        ])
        data_split_test = CIFAR10(args.dataset_dir, train=False, download=True, transform=transform_test)
        test_loader = DataLoader(data_split_test, batch_size=args.batch_size, num_workers=args.num_workers)

        # Optionally add ckpt_path to trainer.predict()
        test_predictions = trainer.predict(model=model, dataloaders=test_loader)

        test_hash_codes = torch.concat([x['hash_codes'] for x in test_predictions])
        test_ground_truths = torch.concat([x['ground_truths'] for x in test_predictions])

        map_score = compute_map_score(train_hash_codes, train_ground_truths, test_hash_codes, test_ground_truths)
        print(f'MAP score: {map_score}')
        mAP_scores.append(float(map_score))

    plot_dir = Path(args.model_ckpt).parent.parent.joinpath('plots')
    plot_dir.mkdir(exist_ok=True, parents=True)
    plot_data = {'std_dev': std_devs, 'mAP_score': mAP_scores}
    with open(plot_dir.joinpath(f'{args.noise_type}.json'), 'w+') as f:
        json.dump(plot_data, f, indent=2)

    plt.figure()
    plt.plot(std_devs, mAP_scores)
    plt.xlabel('std_dev')
    plt.ylabel('mAP_score')
    plt.title('mAP score on CIFAR-10')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir.joinpath(f'{args.noise_type}.png'))


if __name__ == '__main__':
    predict_with_noise()
