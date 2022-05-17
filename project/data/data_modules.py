import random
from typing import Tuple, Any

import torch
from PIL import Image
from torchvision.datasets import CIFAR10


class PairwiseCIFAR10(CIFAR10):
    def __init__(self, root: str, *args, **kwargs):
        super().__init__(root, *args, **kwargs)
        self._make_pairwise()

    def _make_pairwise(self):
        self.classwise_indices = dict()
        for x in set(self.targets):
            self.classwise_indices[x] = {int(y) for y in torch.where(torch.tensor(self.targets) == x)[0]}

        self.image_id_to_class_id = dict()
        for class_id, image_ids in self.classwise_indices.items():
            self.image_id_to_class_id.update({x: class_id for x in image_ids})

        self.class_ids = set(self.classwise_indices)

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image1, image2, target) where target is index of the target class.
        """

        index1 = index
        target1 = self.image_id_to_class_id[index1]
        index2 = random.randint(0, len(self.data) - 1)
        target2 = self.image_id_to_class_id[index2]

        # index1 = math.floor(index / 2)
        # # pick samples from same class
        # if index % 2 == 0:
        #     index2 = random.sample(self.classwise_indices[class_id1].difference({index1}), k=1)[0]
        #     target = 1
        # # pick samples from different class
        # else:
        #     class_id2 = random.sample(self.class_ids.difference({class_id1}), k=1)[0]
        #     index2 = random.sample(self.classwise_indices[class_id2], k=1)[0]
        #     target = 0

        img1, img2 = self.data[index1], self.data[index2]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img1, img2 = Image.fromarray(img1), Image.fromarray(img2)

        if self.transform is not None:
            img1, img2 = self.transform(img1), self.transform(img2)

        if self.target_transform is not None:
            target1 = self.target_transform(target1)
            target2 = self.target_transform(target2)

        return img1, img2, target1, target2

    def __len__(self) -> int:
        return len(self.data)