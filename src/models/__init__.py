"""
Модели: базовые архитектуры + Lightning-обёртка.

LitAudioModel — обёртка над ЛЮБОЙ моделью, возвращающей (logits_cat, logits_tgt).
MobileNetAudio, SmallAudioCNN — референсные архитектуры.
"""
from .models_audio import MobileNetAudio, SmallAudioCNN, DepthwiseSeparableConv
from .lit_model import LitAudioModel

__all__ = [
    "MobileNetAudio",
    "SmallAudioCNN",
    "DepthwiseSeparableConv",
    "LitAudioModel",
]
