"""
Пайплайн для сырого аудио.

Аугментации применяются к волне (audiomentations), затем считается
мел-спектрограмма на GPU. Это физически корректно, но медленно:
первый запуск грузит все аудио в RAM и считает спектрограммы.

Публичный API совпадает с DataPipelineDone, чтобы в ноутбуке
менялась одна строка:

    dm = DataPipelineRaw(...)   # или DataPipelineDone(...)
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
from .encoders import fit_or_load_encoders, to_str_array
from .datasets.raw import AudioDatasetRaw


class DataPipelineRaw(L.LightningDataModule):
    """
    LightningDataModule для сырого аудио.

    Parameters
    ----------
    data_path : str
        Путь к .npz с ключами ``paths``, ``labels_category``, ``labels_target``.
    test_size, val_size : float
        Доли test и val при сплите (0..1).
    seed : int
        Seed для сплита и энкодеров.
    target_encoder_path, category_encoder_path : str | None
        Пути к сохранённым .pkl энкодерам. Если None или файлы не подходят —
        энкодеры обучаются заново и сохраняются в ``encoders/``.
    test_only : bool
        Если True — создаётся только test-датасет (для инференса).
    mel_mean, mel_std : float
        Параметры нормализации мел-спектрограмм.
    batch_size : int
        Размер батча.
    transform : callable | None
        Аугментации из audiomentations (применяются к волне).
    device : str
        Устройство для мел-трансформаций ('cuda' или 'cpu').
    data_shape : tuple[int, int]
        Целевая форма спектрограммы (n_mels, time).
    num_workers : int
        Воркеры DataLoader. Внимание: при num_workers>0 каждый воркер
        скопирует кэш аудио в RAM. Оставляйте 0, если кэш большой.
    """

    def __init__(
        self,
        data_path: str,
        test_size: float,
        val_size: float,
        seed: int,
        batch_size: int,
        mel_mean: float,
        mel_std: float,
        transform=None,
        target_encoder_path: Optional[str] = None,
        category_encoder_path: Optional[str] = None,
        test_only: bool = False,
        device: str = "cuda",
        data_shape: tuple[int, int] = (256, 173),
        num_workers: int = 0,
    ):
        super().__init__()
        # save_hyperparameters логирует всё (кроме transform) в MLflow автоматически
        self.save_hyperparameters(ignore=["transform"])

        self.data_path = data_path
        self.test_size = test_size
        self.val_size = val_size
        self.seed = seed
        self.batch_size = batch_size
        self.mel_mean = mel_mean
        self.mel_std = mel_std
        self.transform = transform
        self.target_encoder_path = target_encoder_path
        self.category_encoder_path = category_encoder_path
        self.test_only = test_only
        self.device = device
        self.data_shape = data_shape
        self.num_workers = num_workers

        # Заполняются в setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # Метаданные (энкодеры, словари) - полезны снаружи
        self.category_to_idx = None
        self.target_to_idx = None
        self.idx_to_category = None
        self.idx_to_target = None
        self.num_categories = None
        self.num_targets = None

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def prepare_data(self):
        """
        Вызывается один раз, на одном процессе.
        Идеально для скачивания/проверки данных. У нас данные уже на диске,
        поэтому только проверяем существование файла.
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Датасет не найден: {self.data_path}")

    def setup(self, stage: Optional[str] = None):
        """
        Вызывается на каждом процессе/устройстве.
        Защита от повторной загрузки: если датасеты уже созданы — выходим.
        """
        if self.train_dataset is not None or self.test_dataset is not None:
            return

        # 1) Загрузка .npz
        paths, labels_category, labels_target = self._load_array(self.data_path)

        # 2) Сплит
        (train, train_cat, train_tgt), \
        (val, val_cat, val_tgt), \
        (test, test_cat, test_tgt) = split_data(
            paths, labels_category, labels_target,
            test_size=self.test_size, val_size=self.val_size, seed=self.seed,
        )

        # 3) Энкодеры
        (enc_cat, enc_tgt,
         self.category_to_idx, self.target_to_idx,
         self.idx_to_category, self.idx_to_target) = fit_or_load_encoders(
            train_category=to_str_array(train_cat),
            train_target=to_str_array(train_tgt),
            category_path=self.category_encoder_path,
            target_path=self.target_encoder_path,
        )
        self.num_categories = len(self.category_to_idx)
        self.num_targets = len(self.target_to_idx)

        # 4) Кодирование
        train_cat_enc = enc_cat.transform(to_str_array(train_cat))
        train_tgt_enc = enc_tgt.transform(to_str_array(train_tgt))
        val_cat_enc = enc_cat.transform(to_str_array(val_cat))
        val_tgt_enc = enc_tgt.transform(to_str_array(val_tgt))
        test_cat_enc = enc_cat.transform(to_str_array(test_cat))
        test_tgt_enc = enc_tgt.transform(to_str_array(test_tgt))

        # 5) Датасеты
        print("Создание датасетов (raw):")
        self.test_dataset = AudioDatasetRaw(
            test, test_cat_enc, test_tgt_enc,
            data_shape=self.data_shape,
            transform=None,                 # test без аугментаций
            normalize=True,
            mel_mean=self.mel_mean,
            mel_std=self.mel_std,
            device=self.device,
        )
        print(f"  - test: {len(self.test_dataset)}")

        if self.test_only:
            self.train_dataset = None
            self.val_dataset = None
            print("  - train/val пропущены (test_only=True)")
            return

        self.train_dataset = AudioDatasetRaw(
            train, train_cat_enc, train_tgt_enc,
            data_shape=self.data_shape,
            transform=self.transform,       # аугментации на волне
            normalize=True,
            mel_mean=self.mel_mean,
            mel_std=self.mel_std,
            device=self.device,
        )
        self.val_dataset = AudioDatasetRaw(
            val, val_cat_enc, val_tgt_enc,
            data_shape=self.data_shape,
            transform=None,                 # val без аугментаций
            normalize=True,
            mel_mean=self.mel_mean,
            mel_std=self.mel_std,
            device=self.device,
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
        """Загружает .npz и возвращает (paths, labels_category, labels_target)."""
        print(f"Загрузка данных из {data_path}...")
        dataset = np.load(data_path, allow_pickle=True)
        paths = dataset["paths"]
        labels_category = dataset["labels_category"]
        labels_target = dataset["labels_target"]
        print(f"  paths: {len(paths)} | "
              f"category: {labels_category.shape} | "
              f"target: {labels_target.shape}")
        return paths, labels_category, labels_target