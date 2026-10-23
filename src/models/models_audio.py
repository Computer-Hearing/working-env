"""
Базовые модели для экспериментов.

Здесь лежат готовые, проверенные архитектуры — как отправная точка
и как референс для сравнения. Студент может:
  1. Использовать их как есть.
  2. Наследоваться от них и переопределять forward.
  3. Писать свою модель в ноутбуке — тогда этот файл не нужен.
"""
from __future__ import annotations

import torch
import torch.nn as nn


# ======================================================================
# Блоки
# ======================================================================

class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Conv с параллельной факторизацией 3x3 -> (1x3 || 3x1).

    Входные каналы делятся пополам между ветвями:
      - первая половина проходит через depthwise 1x3,
      - вторая половина — через depthwise 3x1.
    После конкатенации снова C каналов, затем pointwise 1x1
    (здесь же — downsampling через stride).

    Параметры для C -> C_out:
      depthwise: 3*(C/2) + 3*(C/2) = 3C
      pointwise: C * C_out
    Против 9C + C*C_out у обычной DW 3x3 + PW.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        assert in_channels % 2 == 0, "in_channels должно быть чётным"

        half = in_channels // 2

        # Ветвь 1: depthwise 1x3 на первой половине каналов
        self.dw_h = nn.Conv2d(
            half, half,
            kernel_size=(1, 3),
            stride=1,
            padding=(0, 1),
            groups=half,
            bias=False,
        )
        # Ветвь 2: depthwise 3x1 на второй половине каналов
        self.dw_v = nn.Conv2d(
            half, half,
            kernel_size=(3, 1),
            stride=1,
            padding=(1, 0),
            groups=half,
            bias=False,
        )

        # После cat снова in_channels каналов
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

        # Pointwise: здесь и только здесь делаем downsampling
        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1, stride=stride, padding=0, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x_h, x_v = x.chunk(2, dim=1)   # (B, C/2, H, W) каждая

        x_h = self.dw_h(x_h)           # (B, C/2, H, W)
        x_v = self.dw_v(x_v)           # (B, C/2, H, W)

        x = torch.cat([x_h, x_v], dim=1)  # (B, C, H, W)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.pointwise(x)             # (B, C_out, H/stride, W/stride)
        x = self.bn2(x)
        x = self.relu(x)
        return x


# ======================================================================
# Модели
# ======================================================================

class MobileNetAudio(nn.Module):
    """
    Двухголовая аудио-модель на DepthwiseSeparableConv.

    Parameters
    ----------
    in_channels : int
        Каналы входа (1 для моно-спектрограммы).
    num_categories, num_targets : int
        Размеры голов.
    width_multiplier : float
        Множитель ширины (0.5, 0.75, 1.0, ...).
    config : list[tuple[int, int]] | None
        Список (out_channels, stride) для блоков.
        По умолчанию — типичный MobileNet-конфиг.
    dropout : float
        Dropout перед головами.
    """

    DEFAULT_CONFIG = [
        (32, 2),
        (64, 2),
        (128, 2),
        (256, 2),
        (512, 2),
    ]

    def __init__(
        self,
        in_channels: int = 1,
        num_categories: int = 5,
        num_targets: int = 7,
        width_multiplier: float = 1.0,
        config: list[tuple[int, int]] | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        config = config or self.DEFAULT_CONFIG

        def apply_width(ch: int) -> int:
            # Округляем до чётного (DepthwiseSeparableConv требует чётное)
            v = max(2, int(ch * width_multiplier))
            return v if v % 2 == 0 else v + 1

        # Входной слой
        initial_ch = apply_width(32)
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, initial_ch, kernel_size=3, stride=2,
                      padding=1, bias=False),
            nn.BatchNorm2d(initial_ch),
            nn.ReLU(inplace=True),
        )

        # Блоки
        blocks = []
        in_ch = initial_ch
        for out_ch, stride in config:
            out_ch = apply_width(out_ch)
            blocks.append(DepthwiseSeparableConv(in_ch, out_ch, stride=stride))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)
        self.last_channels = in_ch

        # Пул + головы
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.head_category = nn.Linear(self.last_channels, num_categories)
        self.head_target = nn.Linear(self.last_channels, num_targets)

    def forward(self, x):
        x = self.initial(x)
        x = self.blocks(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.dropout(x)

        category = self.head_category(x)
        target = self.head_target(x)
        return category, target


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
