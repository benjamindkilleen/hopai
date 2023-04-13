import torch
import torch.nn as nn
from torchvision.datasets import MNIST
import logging
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


class MyDataset:
    def __init__(self, train: bool = True) -> None:
        self.mnist = MNIST(root="data", train=train, download=True)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a single item from the dataset.

        Args:
            index (int): The index of the item to return.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The image and label.
        """

        img, label = self.mnist[index]
        img = np.array(img)
        img = img / 255.0
        img = (img - 0.5) * 2
        # augmentation on img, label

        img = torch.from_numpy(img).float()
        img = img.unsqueeze(0)
        label = torch.tensor(label).long()
        return img, label

    def __len__(self) -> int:
        return len(self.mnist)
