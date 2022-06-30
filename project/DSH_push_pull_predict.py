import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset
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
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class CIFAR10C(Dataset):
    def __init__(self, root_dir, num_splits, transform=None):
        super(CIFAR10C, self).__init__()
        corruptions = sorted(root_dir.glob('*'))
        self._corruptions = {x.stem: np.split(np.load(x), num_splits) for x in corruptions if x.stem != 'labels'}
        self._labels = np.split(np.load(root_dir.joinpath('labels.npy')), num_splits)
        self._corruption_type = 'gaussian_noise'  # This is the default value which can be overridden
        self.transform = transform
        self._all_corruption_types = set(self._corruptions.keys())
        self.val_corruption_types = {'speckle_noise', 'gaussian_blur', 'spatter', 'saturate'}
        self.test_corruption_types = self._all_corruption_types.difference(self.val_corruption_types)
        self._severity_level = 0

    @property
    def corruption_type(self):
        return self._corruption_type

    @corruption_type.setter
    def corruption_type(self, value):
        assert value in self._all_corruption_types, 'Invalid corruption type'
        self._corruption_type = value

    @property
    def severity_level(self):
        return self._severity_level

    @severity_level.setter
    def severity_level(self, value):
        assert value in {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
        self._severity_level = value

    def __getitem__(self, index):
        img = self.transform(self._corruptions[self._corruption_type][self._severity_level][index])
        label = self._labels[self._severity_level][index]
        return img, label

    def __len__(self):
        return len(self._labels[self._severity_level])


def comparison_with_baseline(mAP_scores, mAP_scores_baseline):
    mCE, relative_mCE = 100, 100

    assert mAP_scores.get('clean', False), 'results on clean images not present'
    assert mAP_scores_baseline.get('clean', False), 'results on clean images not present'

    categories = sorted(mAP_scores.keys())
    categories.remove('clean')

    # Compute the error
    error_classifier = 1 - torch.Tensor([mAP_scores[x] for x in categories])
    error_baseline = 1 - torch.Tensor([mAP_scores_baseline[x] for x in categories])

    # Corruption Error (CE) and it's mean
    CE = torch.sum(error_classifier, dim=1) / torch.sum(error_baseline, dim=1)
    mCE = torch.mean(CE)

    # Relative CE and it's mean
    relative_CE = torch.sum(error_classifier - mAP_scores['clean'], dim=1) / \
                  torch.sum(error_baseline - mAP_scores_baseline['clean'], dim=1)
    relative_mCE = torch.mean(relative_CE)

    # converting all tensors to floats
    CE = {x: float(CE[idx]) for idx, x in enumerate(categories)}
    mCE = float(mCE)
    relative_CE = {x: float(relative_CE[idx]) for idx, x in enumerate(categories)}
    relative_mCE = float(relative_mCE)

    return CE, mCE, relative_CE, relative_mCE


def predict_with_noise():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--dataset_dir', default=r'/data/p288722/datasets/cifar', type=str)
    parser.add_argument('--num_workers', default=2, type=int,
                        help='how many subprocesses to use for data loading. ``0`` means that the data will be '
                             'loaded in the main process. (default: ``2``)')
    parser.add_argument('--model_ckpt', type=str, required=True)
    parser.add_argument('--corruption_type', default=None, type=str,
                        choices=['gaussian'])
    parser.add_argument('--use_push_pull', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--baseline_classifier_results_dir', required=True, type=str)

    parser = pl.Trainer.add_argparse_args(parser)
    parser = Classifier.add_model_specific_args(parser)
    args = parser.parse_args()

    if not args.accelerator:
        args.accelerator = 'gpu'

    assert Path(args.dataset_dir).exists(), f'{args.dataset_dir} does not exists!'
    assert Path(args.model_ckpt).exists(), f'{args.model_ckpt} path does not exists!'
    assert Path(args.baseline_classifier_results_dir).parent.exists(), \
        f'{Path(args.baseline_classifier_results_dir).parent} path does not exists!'
    Path(args.baseline_classifier_results_dir).mkdir(exist_ok=True, parents=False)

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

    trainer = pl.Trainer.from_argparse_args(args, gpus=1)
    train_predictions = trainer.predict(model=model, dataloaders=train_loader)
    train_hash_codes = torch.concat([x['hash_codes'] for x in train_predictions])
    train_ground_truths = torch.concat([x['ground_truths'] for x in train_predictions])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    clean_test_dataset = CIFAR10(args.dataset_dir, train=False, download=True, transform=transform_test)
    test_dataloader = DataLoader(clean_test_dataset, args.batch_size, num_workers=args.num_workers)

    mAP_scores = defaultdict(list)

    predictions = trainer.predict(model=model, dataloaders=test_dataloader)
    test_hash_codes = torch.concat([x['hash_codes'] for x in predictions])
    test_ground_truths = torch.concat([x['ground_truths'] for x in predictions])
    map_score = compute_map_score(train_hash_codes, train_ground_truths, test_hash_codes, test_ground_truths)
    mAP_scores['clean'] = float(map_score)

    dataset_name = 'CIFAR-10-C-EnhancedSeverity'
    corrupted_test_dataset = CIFAR10C(Path(args.dataset_dir).joinpath(dataset_name),
                                      num_splits=10, transform=transform_test)
    # fixme: filter the corruption_types based on the input arguments
    corruption_types = corrupted_test_dataset.test_corruption_types

    for corruption_type in corruption_types:
        corrupted_test_dataset.corruption_type = corruption_type
        for severity_level in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            corrupted_test_dataset.severity_level = severity_level

            test_dataloader = DataLoader(corrupted_test_dataset, args.batch_size, num_workers=args.num_workers)
            predictions = trainer.predict(model=model, dataloaders=test_dataloader)
            test_hash_codes = torch.concat([x['hash_codes'] for x in predictions])
            test_ground_truths = torch.concat([x['ground_truths'] for x in predictions])
            map_score = compute_map_score(train_hash_codes, train_ground_truths, test_hash_codes, test_ground_truths)
            mAP_scores[corruption_type].append(float(map_score))

    print(mAP_scores)
    results_dir = Path(args.model_ckpt).parent.parent.joinpath('results')
    results_dir.mkdir(exist_ok=True, parents=True)
    with open(results_dir.joinpath(f'mAP_scores_{dataset_name}.json'), 'w+') as f:
        json.dump(mAP_scores, f, indent=2)
    baseline_results_file = Path(args.baseline_classifier_results_dir).joinpath(f'mAP_scores_{dataset_name}.json')
    with open(baseline_results_file) as f:
        mAP_scores_baseline = json.load(f)
    CE, mCE, relative_CE, relative_mCE = comparison_with_baseline(mAP_scores, mAP_scores_baseline)
    with open(results_dir.joinpath(f'all_scores_{dataset_name}.json'), 'w+') as f:
        json.dump({
            'baseline_results_file': str(baseline_results_file),
            'CE': CE,
            'mCE': mCE,
            'relative_CE': relative_CE,
            'relative_mCE': relative_mCE,
            'mAP': mAP_scores,
        }, f, indent=2)


if __name__ == '__main__':
    predict_with_noise()
