import lightning as L
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from .dataset import AudioDatasetDone
from .constants import C


class DataPipelineMulticlass(L.LightningDataModule):
    def __init__(
            self,
            data_path: str,
            batch_size: int = 32,
            test_size: float = 0.15,
            val_size: float = 0.15,
            seed: int = 42,
            mode: str = "Conv",
            data_key: str | None = "X",
            label_key: str | None = "y",
            mean_key: str | None = None,
            std_key: str | None = None,
            fallback_mean: float = 0.0,
            fallback_std: float = 1.0,
            num_workers: int = 4,
            augment_train: bool = True,
            data_shape: tuple = None,
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.test_size = test_size
        self.val_size = val_size
        self.seed = seed
        self.mode = mode
        self.data_key = data_key
        self.label_key = label_key
        self.mean_key = mean_key
        self.std_key = std_key
        self.fallback_mean = fallback_mean
        self.fallback_std = fallback_std
        self.num_workers = num_workers
        self.augment_train = augment_train
        self.data_shape = data_shape or C.DATA_SHAPE

    def setup(self, stage=None):
        if hasattr(self, 'train_dataset') and self.train_dataset is not None:
            return

        data = np.load(self.data_path, allow_pickle=True)

        # 1. Проверка ключей (с учетом None)
        if self.data_key is None or self.label_key is None:
            raise ValueError("Для загрузки из .npz необходимо указать data_key и label_key.")

        if self.data_key not in data.files:
            raise KeyError(f"Ключ '{self.data_key}' не найден. Доступные: {data.files}")
        if self.label_key not in data.files:
            raise KeyError(f"Ключ '{self.label_key}' не найден. Доступные: {data.files}")

        specs = data[self.data_key]
        labels = data[self.label_key]

        # 2. Обработка one-hot
        if labels.ndim > 1 and labels.shape[1] > 1:
            labels = np.argmax(labels, axis=1)

        # 3. Правильное число классов
        self.num_classes = int(labels.max()) + 1

        # 4. Сплит
        X_train, X_test, y_train, y_test = train_test_split(
            specs, labels, test_size=self.test_size, random_state=self.seed, stratify=labels
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=self.val_size, random_state=self.seed, stratify=y_train
        )

        # 5. Логика нормализации
        if self.mean_key and self.mean_key in data.files:
            self.mel_mean = float(data[self.mean_key])
            print(f"mean взят из npz: {self.mel_mean:.4f}")
        else:
            print("Вычисление mean на train...")
            self.mel_mean = float(np.mean(X_train))

        if self.std_key and self.std_key in data.files:
            self.mel_std = float(data[self.std_key])
            print(f"std взят из npz: {self.mel_std:.4f}")
        else:
            print("Вычисление std на train...")
            self.mel_std = float(np.std(X_train))

        print(f"  Итоговая нормализация: mean={self.mel_mean:.4f}, std={self.mel_std:.4f}")

        # 6. Датасеты
        self.train_dataset = AudioDatasetDone(
            X_train, y_train, mode=self.mode,
            mel_mean=self.mel_mean, mel_std=self.mel_std,
            augment=self.augment_train,
            data_shape=self.data_shape,
        )
        self.val_dataset = AudioDatasetDone(
            X_val, y_val, mode=self.mode,
            mel_mean=self.mel_mean, mel_std=self.mel_std,
            augment=False,
            data_shape=self.data_shape,
        )
        self.test_dataset = AudioDatasetDone(
            X_test, y_test, mode=self.mode,
            mel_mean=self.mel_mean, mel_std=self.mel_std,
            augment=False,
            data_shape=self.data_shape,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=True, persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=True, persistent_workers=True
        )