"""
Dataset для сырого аудио с кэшированием в RAM.

Аугментации применяются к волне (audiomentations), затем считается
мел-спектрограмма на GPU, затем обрезка/паддинг и нормализация.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import torchaudio
import librosa
from torch.utils.data import Dataset


class AudioDatasetRaw(Dataset):
    """
    Dataset с кэшированием аудио в RAM.

    Parameters
    ----------
    audio_data : array-like[str]
        Пути к аудиофайлам.
    categories, targets : array-like[int]
        Закодированные метки (int-индексы).
    data_shape : tuple[int, int]
        Целевая форма спектрограммы (n_mels, time).
    transform : callable | None
        Аугментации из audiomentations (применяются к волне).
    normalize : bool
        Нормализовать ли спектрограмму по mel_mean/mel_std.
    mel_mean, mel_std : float | None
        Параметры нормализации.
    device : str | None
        Устройство для мел-трансформаций. None → автоопределение.
    """

    def __init__(
        self,
        audio_data,
        categories,
        targets,
        data_shape: tuple[int, int] = (256, 173),
        transform=None,
        normalize: bool = False,
        mel_mean: float | None = None,
        mel_std: float | None = None,
        device: str | None = None,
        mode: str | None = 'Default'
    ):
        assert len(audio_data) == len(categories) == len(targets)

        self.audio_data = audio_data
        self.categories = categories
        self.targets = targets
        self.data_shape = data_shape
        self.transform = transform
        self.normalize = normalize
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = mode
        assert mode in ("Conv", "Sequence", "Default", None), f"Неизвестный mode: {mode}, , возможные: 'Conv', 'Sequence', 'Default'"

        self.mean = (
            torch.tensor(mel_mean, dtype=torch.float32, device=self.device)
            if mel_mean is not None else None
        )
        self.std = (
            torch.tensor(mel_std, dtype=torch.float32, device=self.device)
            if mel_std is not None else None
        )

        # Кэшируем всё аудио в RAM
        print(f"Загрузка {len(audio_data)} аудио в RAM...")
        self.audio_cache = []
        for i, path in enumerate(audio_data):
            y, sr = librosa.load(path, sr=None)
            self.audio_cache.append((y, sr))
            if (i + 1) % 500 == 0:
                print(f"  Загружено {i + 1}/{len(audio_data)}")
        print(f"  Готово: {len(self.audio_cache)} файлов в RAM")

        # Mel-трансформации на устройстве
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=44000,
            n_fft=2048,
            hop_length=512,
            n_mels=256,
            f_max=16384,
        ).to(self.device)

        self.db_transform = torchaudio.transforms.AmplitudeToDB(
            top_db=80.0
        ).to(self.device)

    def __len__(self):
        return len(self.audio_data)

    def __getitem__(self, idx):
        y, sr = self.audio_cache[idx]

        # Аугментации на CPU (audiomentations работает с numpy)
        if self.transform is not None:
            y = self.transform(samples=y, sample_rate=sr)

        # Спектрограмма на устройстве
        y_tensor = torch.as_tensor(y, dtype=torch.float32).to(self.device)
        mel_spec = self.mel_transform(y_tensor)      # [n_mels, time]
        mel_spec_db = self.db_transform(mel_spec)    # [n_mels, time]

        # Обрезка/дополнение до фиксированного размера
        spectrogram = self._truncate_spectrogram(mel_spec_db)

        # Нормализация
        if self.normalize and self.mean is not None and self.std is not None:
            spectrogram = (spectrogram - self.mean) / (self.std + 1e-8)

        # Трансформация спектрограммы в зависимости от типа модели
        if self.mode == "Conv":                                   # (1, n_mels, time)
            spectrogram = spectrogram.unsqueeze(0).float().cpu()
        elif self.mode == 'Sequence':                             # (time, n_mels)
            spectrogram = spectrogram.transpose(0, 1)
        elif self.mode == 'Default' or self.mode is None:
            pass
        else:
            raise ValueError(f"Неизвестный mode: {self.mode}, возможные: 'Conv', 'Sequence', 'Default'")

        return (
            spectrogram,
            self.categories[idx],
            self.targets[idx],
        )

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