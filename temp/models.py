"""
Fixed model scaffold.

What is fixed:
- the overall Part A backbone depth,
- the channel sizes,
- the pooling pattern,
- the compact student architecture used for distillation.

What students must implement:
- the depthwise separable block,
- the `SeparableCNN` and `CompactSeparableCNN` paths that use that block.
"""

from typing import Optional

import torch
from torch import nn


class StandardCNN(nn.Module):
    """
    Fixed reference CNN for Part A.

    Students should use this implementation as given and compare it against
    their completed separable version under the same architecture budget.
    """

    def __init__(
        self,
        num_classes: int = 10,
        embedding_dim: Optional[int] = None,
        input_channels: int = 1,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        output_dim = embedding_dim or num_classes
        self.head = nn.Linear(128, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.head(x)


class DepthwiseSeparableBlock(nn.Module):
    """
    TODO for students:
    implement one depthwise separable block using a depthwise 3x3 convolution
    followed by a pointwise 1x1 convolution.

    Requirements:
    - use `stride=stride` in the depthwise convolution,
    - keep `padding=1` for the 3x3 depthwise convolution,
    - use `groups=in_channels` for the depthwise convolution,
    - keep BatchNorm + ReLU after each convolution stage.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Identity()
        raise NotImplementedError("Implement DepthwiseSeparableBlock before training SeparableCNN.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SeparableCNN(nn.Module):
    """
    Fixed scaffold for the Part A separable backbone.

    Students must keep this macro-architecture unchanged and only complete the
    missing separable building block implementation.
    """

    def __init__(
        self,
        num_classes: int = 10,
        embedding_dim: Optional[int] = None,
        input_channels: int = 1,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.features = nn.Sequential(
            DepthwiseSeparableBlock(32, 64, stride=2),
            DepthwiseSeparableBlock(64, 128, stride=2),
            DepthwiseSeparableBlock(128, 128, stride=1),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        output_dim = embedding_dim or num_classes
        self.head = nn.Linear(128, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.features(x)
        x = x.flatten(1)
        return self.head(x)


class CompactSeparableCNN(nn.Module):
    """
    Fixed smaller student architecture for compression.

    This is not a free-design component. Once `DepthwiseSeparableBlock` is
    implemented, this model should work without further architecture changes.
    """

    def __init__(
        self,
        num_classes: int = 10,
        embedding_dim: Optional[int] = None,
        input_channels: int = 1,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.features = nn.Sequential(
            DepthwiseSeparableBlock(16, 32, stride=2),
            DepthwiseSeparableBlock(32, 64, stride=2),
            DepthwiseSeparableBlock(64, 64, stride=1),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        output_dim = embedding_dim or num_classes
        self.head = nn.Linear(64, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.features(x)
        x = x.flatten(1)
        return self.head(x)


def build_model(
    name: str,
    num_classes: int = 10,
    embedding_dim: Optional[int] = None,
    input_channels: int = 1,
) -> nn.Module:
    if name == "cnn":
        return StandardCNN(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            input_channels=input_channels,
        )
    if name == "separable_cnn":
        return SeparableCNN(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            input_channels=input_channels,
        )
    if name == "compact_separable_cnn":
        return CompactSeparableCNN(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            input_channels=input_channels,
        )
    raise ValueError(f"Unknown model: {name}")
