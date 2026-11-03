import lightning as L
from lightning.pytorch.loggers import CSVLogger, MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from .config import env_config

class Runner:
    def __init__(self, experiment_name: str = "default", use_mlflow: bool = False):
        self.experiment_name = experiment_name
        self.use_mlflow = use_mlflow
        env_config.setup_dirs()

    def get_trainer(
        self, 
        max_epochs: int = 10, 
        monitor: str = "val_loss", 
        mode: str = "min", 
        patience: int = 10, 
        **trainer_kwargs
    ) -> L.Trainer:
        
        # Логгер (если не нужен MLflow, используем CSVLogger, он быстрее)
        if self.use_mlflow:
            logger = MLFlowLogger(
                experiment_name=self.experiment_name,
                tracking_uri=env_config.mlflow_tracking_uri
            )
        else:
            logger = CSVLogger(save_dir=env_config.logs_dir, name=self.experiment_name)

        # 2. Коллбэки
        callbacks = [
            ModelCheckpoint(
                dirpath=f"{env_config.checkpoints_dir}/{self.experiment_name}",
                filename="best-{epoch:02d}-{" + monitor + ":.3f}",
                monitor=monitor,
                mode=mode,
                save_top_k=1,
                verbose=False, # Отключаем лишний вывод
            ),
            EarlyStopping(
                monitor=monitor,
                mode=mode,
                patience=patience,
                verbose=False,
            ),
            LearningRateMonitor(logging_interval="epoch"),
        ]

        # Собираем Trainer с оптимизациями
        final_kwargs = {
            "max_epochs": max_epochs,
            "accelerator": "auto",
            "devices": 1,
            "logger": logger,
            "callbacks": callbacks,
            "log_every_n_steps": 50, 
            "check_val_every_n_epoch": 1,
            "enable_progress_bar": True,
            "enable_model_summary": False,
            "num_sanity_val_steps": 0,
            **trainer_kwargs
        }

        return L.Trainer(**final_kwargs)