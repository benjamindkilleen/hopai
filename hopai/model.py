import torch
import torch.nn as nn
import logging
import numpy as np

log = logging.getLogger(__name__)


class MyModel(nn.Module):
    def __init__(self, levels: int = 3, blocks: int = 1, channels: int = 32) -> None:
        super().__init__()

        self.encoder = nn.ModuleList()

        input_channels = 1
        for l in range(levels):
            for _ in range(blocks):
                output_channels = channels * (2**l)
                self.encoder.append(
                    nn.Sequential(
                        nn.Conv2d(input_channels, output_channels, 3, 1, 1),
                        nn.BatchNorm2d(output_channels),
                        nn.ReLU(),
                    )
                )
                input_channels = output_channels
            self.encoder.append(nn.MaxPool2d(2))
            self.encoder.append(nn.Dropout(0.2))

        self.encoder.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.encoder.append(nn.Flatten())

        self.head = nn.Sequential(
            nn.Linear(output_channels, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.encoder:
            x = layer(x)
        x = self.head(x)
        return x
