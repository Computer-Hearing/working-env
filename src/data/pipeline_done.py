"""
Пайплайн для готовых мел-спектрограмм.

Ожидает .npz с ключами:
  - array1 : спектрограммы, shape (N, 256, 173) или (N, 1, 256, 173)
  - array2 : one-hot категории, shape (N, 5)
  - array3 : one-hot таргеты,   shape (N, 7)

Аугментации — SpecAugment (time/freq masking, gain, noise) прямо над
спектрограммой. Дешево по вычислениям, но менее физически осмысленно,
чем аугментации над волной в DataPipelineRaw.

Публичный API совпадает с DataPipelineRaw:

    dm = DataPipelineDone(...)   # или DataPipelineRaw(...)
    trainer.fit(lit_model, datamodule=dm)
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
import lightning as L

from .spliter import split_data
from .datasets.done import AudioDatasetDone


class DataPipelineDone(L.LightningDataModule):
    """
    LightningDataModule для готовых мел-спектрограмм.

    Parameters
    ----------
    data_path : str
        Путь к .npz с ключами ``array1``, ``array2``, ``array3``.
    test_size, val_size : float
        Доли test и val при сплите (0..1).
    seed : int
        Seed для сплита.
    batch_size : int
        Размер батча.
    test_only : bool
        Если True — создаётся только test-датасет (для инференса).
    mode : str
        'CNN' (добавляет канал) или 'GRU' (транспонирует время/частоту).
    augment_train : bool
        Включать ли SpecAugment на train.
    time_mask_max, freq_mask_max : int
        Максимальная ширина/высота маски.
    n_masks : int
        Сколько масок каждого типа применять.
    gain_range : tuple[float, float]
        Диапазон случайного усиления.
    noise_sigma : float
        Стд гауссова шума (0 — выключено).
    mel_mean, mel_std : float
        Параметры нормализации спектрограмм.
    num_workers : int
        Воркеры DataLoader.
    """

    def __init__(
        self,
        data_path: str,
        test_size: float,
        val_size: float,
        seed: int,
        batch_size: int,
        mode: str = "CNN",
        test_only: bool = False,
        # --- SpecAugment ---
        augment_train: bool = True,
        time_mask_max: int = 20,
        freq_mask_max: int = 20,
        n_masks: int = 2,
        gain_range: tuple[float, float] = (0.8, 1.2),
        noise_sigma: float = 0.0,
        # --- нормализация ---
        mel_mean: float = -41.9199,
        mel_std: float = 17.9253,
        num_workers: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.data_path = data_path
        self.test_size = test_size
        self.val_size = val_size
        self.seed = seed
        self.batch_size = batch_size
        self.mode = mode
        self.test_only = test_only

        self.augment_train = augment_train
        self.time_mask_max = time_mask_max
        self.freq_mask_max = freq_mask_max
        self.n_masks = n_masks
        self.gain_range = gain_range
        self.noise_sigma = noise_sigma

        self.mel_mean = mel_mean
        self.mel_std = mel_std
        self.num_workers = num_workers

        # Заполняются в setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # Метаданные для удобства снаружи
        self.num_categories = None
        self.num_targets = None

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def prepare_data(self):
        """Вызывается один раз, на одном процессе. Только проверка файла."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Датасет не найден: {self.data_path}")

    def setup(self, stage: Optional[str] = None):
        """
        Вызывается на каждом процессе. Защита от повторной загрузки:
        если датасеты уже созданы — выходим.
        """
        if self.train_dataset is not None or self.test_dataset is not None:
            return

        # 1) Загрузка .npz
        data, labels_category, labels_target = self._load_array(self.data_path)

        # 2) Сплит
        (train, train_cat, train_tgt), \
        (val, val_cat, val_tgt), \
        (test, test_cat, test_tgt) = split_data(
            data, labels_category, labels_target,
            test_size=self.test_size, val_size=self.val_size, seed=self.seed,
        )

        # 3) Число классов (для LitMobileNetAudio / torchmetrics)
        self.num_categories = labels_category.shape[1]
        self.num_targets = labels_target.shape[1]

        # 4) Датасеты
        print("Создание датасетов (done):")

        self.test_dataset = AudioDatasetDone(
            test, test_cat, test_tgt,
            mode=self.mode,
            mel_mean=self.mel_mean,
            mel_std=self.mel_std,
            augment=False,                  # test без аугментаций
        )
        print(f"  - test: {len(self.test_dataset)}")

        if self.test_only:
            self.train_dataset = None
            self.val_dataset = None
            print("  - train/val пропущены (test_only=True)")
            return

        self.train_dataset = AudioDatasetDone(
            train, train_cat, train_tgt,
            mode=self.mode,
            mel_mean=self.mel_mean,
            mel_std=self.mel_std,
            augment=self.augment_train,
            time_mask_max=self.time_mask_max,
            freq_mask_max=self.freq_mask_max,
            n_masks=self.n_masks,
            gain_range=self.gain_range,
            noise_sigma=self.noise_sigma,
        )
        self.val_dataset = AudioDatasetDone(
            val, val_cat, val_tgt,
            mode=self.mode,
            mel_mean=self.mel_mean,
            mel_std=self.mel_std,
            augment=False,                  # val без аугментаций
        )
        print(f"  - train: {len(self.train_dataset)}")
        print(f"  - val:   {len(self.val_dataset)}")

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------

    def train_dataloader(self):
        if self.train_dataset is None:
            return None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    # ------------------------------------------------------------------
    # Внутренние хелперы
    # ------------------------------------------------------------------

    @staticmethod
    def _load_array(data_path: str):
        """Загружает .npz и возвращает (data, labels_category, labels_target)."""
        print(f"Загрузка данных из {data_path}...")
        dataset = np.load(data_path)

        data = dataset["array1"]
        labels_category = dataset["array2"]
        labels_target = dataset["array3"]

        print(f"  Спектрограммы: {data.shape}, dtype={data.dtype}")
        print(f"  Категории:     {labels_category.shape}, dtype={labels_category.dtype}")
        print(f"  Таргеты:       {labels_target.shape}, dtype={labels_target.dtype}")

        return data, labels_category, labels_target