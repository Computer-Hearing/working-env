from __future__ import annotations

import torch
import torch.nn as nn


class SmallAudioCNN(nn.Module):
    """
    Маленькая свёрточная модель для быстрых экспериментов / smoke-тестов.
    Не претендует на качество - нужна, чтобы проверить, что пайплайн работает.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_categories: int = 5,
        num_targets: int = 7,
        base_channels: int = 16,
    ):
        super().__init__()

        def block(c_in, c_out, stride):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
            )

        self.features = nn.Sequential(
            block(in_channels, base_channels, 2),
            block(base_channels, base_channels * 2, 2),
            block(base_channels * 2, base_channels * 4, 2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head_category = nn.Linear(base_channels * 4, num_categories)
        self.head_target = nn.Linear(base_channels * 4, num_targets)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.head_category(x), self.head_target(x)