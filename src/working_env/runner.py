import lightning as L
from .config import env_config
from .tracking import get_logger
from .callbacks import get_default_callbacks


class Runner:
    """
    Универсальный сборщик Trainer.
    Использует глобальные настройки из env_config (пути, сид и т.д.)
    """

    def __init__(self, experiment_name: str = "default", use_mlflow: bool = None):
        self.experiment_name = experiment_name
        self.use_mlflow = use_mlflow
        env_config.setup_dirs()  # Создаем папки logs, checkpoints, reports

    def get_trainer(
            self,
            max_epochs: int = 10,
            monitor: str = "val_loss",
            mode: str = "min",
            patience: int = 10,
            **trainer_kwargs
    ) -> L.Trainer:
        """
        Собирает Lightning Trainer.

        Parameters:
        -----------
        max_epochs : int
            Количество эпох.
        monitor : str
            Метрика для EarlyStopping и ModelCheckpoint.
        mode : str
            'min' или 'max'.
        patience : int
            Терпение для EarlyStopping.
        **trainer_kwargs :
            Любые другие аргументы Lightning Trainer (например, precision, gradient_clip_val).
        """
        logger = get_logger(
            experiment_name=self.experiment_name,
            use_mlflow=self.use_mlflow
        )

        callbacks = get_default_callbacks(
            experiment_name=self.experiment_name,
            monitor=monitor,
            mode=mode,
            patience=patience
        )

        # Мержим глобальные настройки с переданными аргументами
        final_kwargs = {
            "max_epochs": max_epochs,
            "accelerator": env_config.accelerator,
            "devices": env_config.devices,
            "log_every_n_steps": env_config.log_every_n_steps,
            "logger": logger,
            "callbacks": callbacks,
            **trainer_kwargs  # kwargs пользователя переопределяют дефолты
        }

        return L.Trainer(**final_kwargs)