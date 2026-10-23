"""
Датасеты: raw (сырое аудио + аугментации на волне) и done (готовые спектрограммы).
"""
from .raw import AudioDatasetRaw
from .done import AudioDatasetDone

__all__ = ["AudioDatasetRaw", "AudioDatasetDone"]