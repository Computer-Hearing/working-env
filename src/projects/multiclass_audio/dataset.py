import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class AudioDatasetDone(Dataset):
    def __init__(
        self,
        spectrograms,
        labels,
        mode="Conv",
        mel_mean=0.0,
        mel_std=1.0,
        augment=False,
        data_shape=(256, 173),
        time_mask_max=20,
        freq_mask_max=20,
        n_masks=2,
        gain_range=(0.8, 1.2),
        noise_sigma=0.0,
    ):
        # Данные остаются на CPU (Lightning сам перенесет батчи на GPU)
        self.spectrograms = torch.as_tensor(spectrograms, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        
        self.mode = mode
        self.augment = augment
        self.data_shape = data_shape
        
        # mean и std тоже на CPU
        self.mel_mean = torch.tensor(mel_mean, dtype=torch.float32)
        self.mel_std = torch.tensor(mel_std, dtype=torch.float32)
        
        # Параметры аугментации
        self.time_mask_max = time_mask_max
        self.freq_mask_max = freq_mask_max
        self.n_masks = n_masks
        self.gain_range = gain_range
        self.noise_sigma = noise_sigma

    def __len__(self):
        return len(self.spectrograms)

    def __getitem__(self, idx):
        spec = self.spectrograms[idx]
        
        # Truncate/Pad до фиксированного размера
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

        # 4. Подготовка формы
        if self.mode == "Conv":
            spec = spec.unsqueeze(0)  # (1, n_mels, time)
        elif self.mode == "Sequence":
            spec = spec.transpose(0, 1)  # (time, n_mels)

        return spec, self.labels[idx]

    # ------------------------------------------------------------------
    # Truncate / Pad
    # ------------------------------------------------------------------
    def _truncate_spectrogram(self, spectrogram):
        """Приводит спектрограмму к фиксированному размеру (n_mels, time)."""
        target_w = self.data_shape[1]
        current_w = spectrogram.shape[-1]

        if current_w > target_w:
            return spectrogram[..., :target_w]
        elif current_w < target_w:
            pad = target_w - current_w
            return F.pad(spectrogram, (0, pad), mode="constant", value=0.0)
        return spectrogram

    # ------------------------------------------------------------------
    # SpecAugment
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