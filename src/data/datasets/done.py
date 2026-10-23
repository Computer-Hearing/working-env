"""
Dataset для готовых мел-спектрограмм с аугментациями SpecAugment.
"""
from __future__ import annotations

import torch
from torch.utils.data import Dataset
import torch.functional as F


class AudioDatasetDone(Dataset):
    """
    Dataset для готовых спектрограмм.

    Parameters
    ----------
    spectrograms : array-like, shape (N, n_mels, time)
        Мел-спектрограммы (уже в dB).
    categories, targets : array-like, shape (N, num_classes)
        One-hot метки.
    mode : str
        'CNN' — добавляет канал (N, 1, n_mels, time).
        'GRU' — транспонирует (N, time, n_mels).
    mel_mean, mel_std : float
        Параметры нормализации.
    augment : bool
        Применять ли SpecAugment.
    time_mask_max, freq_mask_max : int
        Максимальная ширина/высота маски.
    n_masks : int
        Сколько масок каждого типа.
    gain_range : tuple[float, float]
        Диапазон случайного усиления.
    noise_sigma : float
        Стд гауссова шума (0 — выключено).
    """

    def __init__(
        self,
        spectrograms,
        categories,
        targets,
        mode: str = "CNN",
        mel_mean: float = -41.9199,
        mel_std: float = 17.9253,
        augment: bool = False,
        time_mask_max: int = 20,
        freq_mask_max: int = 20,
        n_masks: int = 2,
        data_shape: tuple[int, int] = (256, 173),
        gain_range: tuple[float, float] = (0.8, 1.2),
        noise_sigma: float = 0.0,
    ):
        assert len(spectrograms) == len(categories) == len(targets)
        assert mode in ("Conv", "Sequence", "Default", None), f"Неизвестный mode: {mode}, , возможные: 'Conv', 'Sequence', 'Default'"

        self.spectrograms = torch.as_tensor(spectrograms, dtype=torch.float32)
        self.categories = torch.as_tensor(categories, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)

        self.mode = mode
        self.augment = augment
        self.data_shape = data_shape

        self.mel_mean = torch.tensor(mel_mean, dtype=torch.float32)
        self.mel_std = torch.tensor(mel_std, dtype=torch.float32)

        self.time_mask_max = time_mask_max
        self.freq_mask_max = freq_mask_max
        self.n_masks = n_masks
        self.gain_range = gain_range
        self.noise_sigma = noise_sigma

    def __len__(self):
        return len(self.spectrograms)

    def __getitem__(self, idx):
        spec = self.spectrograms[idx]  # (n_mels, time)
        
        # Обрезка/дополнение до фиксированного размера
        spec = self._truncate_spectrogram(spec)

        # Нормализация
        spec = (spec - self.mel_mean) / (self.mel_std + 1e-8)

        # Аугментации
        if self.augment:
            spec = self._time_mask(spec)
            spec = self._freq_mask(spec)
            spec = self._random_gain(spec)
            if self.noise_sigma > 0:
                spec = spec + torch.randn_like(spec) * self.noise_sigma

        # Смена формы
        if self.mode == "Conv":
            spec = spec.unsqueeze(0)          # (1, n_mels, time)
        elif self.mode == "Sequence":
            spec = spec.transpose(0, 1)       # (time, n_mels)
        elif self.mode == "Default" or self.mode is None:
            pass
        else:
            raise ValueError(f"Неизвестный mode: {self.mode}, возможные: 'Conv', 'Sequence', 'Default'")

        return spec, self.categories[idx], self.targets[idx]

    # ------------------------------------------------------------------
    # Аугментации (SpecAugment)
    # ------------------------------------------------------------------

    def _time_mask(self, spec):
        T = spec.shape[1]
        for _ in range(self.n_masks):
            w = torch.randint(1, self.time_mask_max + 1, (1,)).item()
            if T - w <= 1:
                continue
            t0 = torch.randint(0, T - w, (1,)).item()
            spec[:, t0:t0 + w] = 0.0
        return spec

    def _freq_mask(self, spec):
        F_dim = spec.shape[0]
        for _ in range(self.n_masks):
            h = torch.randint(1, self.freq_mask_max + 1, (1,)).item()
            if F_dim - h <= 1:
                continue
            f0 = torch.randint(0, F_dim - h, (1,)).item()
            spec[f0:f0 + h, :] = 0.0
        return spec

    def _random_gain(self, spec):
        g = torch.empty(1).uniform_(*self.gain_range).item()
        return spec * g
    
    # ------------------------------------------------------------------
    # Утилиты для спектрограммы
    # ------------------------------------------------------------------
     
    def _truncate_spectrogram(self, spectrogram):
        """
        Приводит спектрограмму к фиксированному размеру (тензор):
        - длиннее target_width → обрезает справа,
        - короче → дополняет нулями справа.
        """
        target_width = self.data_shape[1]
        current_width = spectrogram.shape[-1]
    
        if current_width > target_width:
            return spectrogram[..., :target_width]
        if current_width < target_width:
            padding = target_width - current_width
            return F.pad(spectrogram, (0, padding), mode="constant", value=0.0)
        return spectrogram