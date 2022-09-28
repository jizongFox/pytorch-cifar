from multiprocessing.spawn import prepare
from torchvision.datasets import CIFAR10 as _CIFAR10
from typing import Any, Tuple
from PIL import Image


class CIFAR10(_CIFAR10):
    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    @property
    def num_classes(self) -> int:
        return len(self.classes)