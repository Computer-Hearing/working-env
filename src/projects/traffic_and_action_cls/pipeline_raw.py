from __future__ import annotations

import os
from typing import Optional

import numpy as np
from torch.utils.data import DataLoader
import lightning as L

from src.working_env.spliter import split_data
from .encoders import fit_or_load_encoders, to_str_array
from .dataset import AudioDatasetRaw


class DataPipelineRaw(L.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        test_size: float,
        val_size: float,
        seed: int,
        batch_size: int,
        mel_mean: float = -41.9199,
        mel_std: float = 17.9253,
        target_sr: int = 44100,
        transform=None,
        target_encoder_path: Optional[str] = None,
        category_encoder_path: Optional[str] = None,
        test_only: bool = False,
        device: str = "cuda",
        data_shape: tuple[int, int] = (256, 173),
        num_workers: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["transform"])

        self.data_path = data_path
        self.test_size = test_size
        self.val_size = val_size
        self.seed = seed
        self.batch_size = batch_size
        self.mel_mean = mel_mean
        self.mel_std = mel_std
        self.target_sr = target_sr
        self.transform = transform
        self.target_encoder_path = target_encoder_path
        self.category_encoder_path = category_encoder_path
        self.test_only = test_only
        self.device = device
        self.data_shape = data_shape
        self.num_workers = num_workers

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.category_to_idx = None
        self.target_to_idx = None
        self.idx_to_category = None
        self.idx_to_target = None
        self.num_categories = None
        self.num_targets = None

    def prepare_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Датасет не найден: {self.data_path}")

    def setup(self, stage: Optional[str] = None):
        if self.train_dataset is not None or self.test_dataset is not None:
            return

        paths, labels_category, labels_target = self._load_array(self.data_path)

        (train, train_cat, train_tgt), \
        (val, val_cat, val_tgt), \
        (test, test_cat, test_tgt) = split_data(
            paths, labels_category, labels_target,
            test_size=self.test_size, val_size=self.val_size, seed=self.seed,
        )

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

        train_cat_enc = enc_cat.transform(to_str_array(train_cat))
        train_tgt_enc = enc_tgt.transform(to_str_array(train_tgt))
        val_cat_enc = enc_cat.transform(to_str_array(val_cat))
        val_tgt_enc = enc_tgt.transform(to_str_array(val_tgt))
        test_cat_enc = enc_cat.transform(to_str_array(test_cat))
        test_tgt_enc = enc_tgt.transform(to_str_array(test_tgt))

        print("Создание датасетов (raw):")
        self.test_dataset = AudioDatasetRaw(
            test, test_cat_enc, test_tgt_enc,
            target_sr=self.target_sr,
            data_shape=self.data_shape,
            transform=None,
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
            target_sr=self.target_sr,
            data_shape=self.data_shape,
            transform=self.transform,
            device=self.device,
        )
        self.val_dataset = AudioDatasetRaw(
            val, val_cat_enc, val_tgt_enc,
            target_sr=self.target_sr,
            data_shape=self.data_shape,
            transform=None,
            device=self.device,
        )
        print(f"  - train: {len(self.train_dataset)}")
        print(f"  - val:   {len(self.val_dataset)}")

    def train_dataloader(self):
        if self.train_dataset is None: return None
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        if self.val_dataset is None: return None
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    @staticmethod
    def _load_array(data_path: str):
        print(f"Загрузка данных из {data_path}...")
        dataset = np.load(data_path, allow_pickle=True)
        return dataset["paths"], dataset["labels_category"], dataset["labels_target"]
