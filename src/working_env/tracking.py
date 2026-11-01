from lightning.pytorch.loggers import MLFlowLogger, CSVLogger
from .config import env_config

def get_logger(experiment_name: str = "default", use_mlflow: bool = None):
    """
    Возвращает логгер для Trainer.
    Если use_mlflow не передан явно, берется значение из глобального env_config.
    """
    if use_mlflow is None:
        use_mlflow = env_config.use_mlflow

    if use_mlflow:
        return MLFlowLogger(
            experiment_name=experiment_name,
            tracking_uri=env_config.mlflow_tracking_uri
        )
    else:
        return CSVLogger(
            save_dir=env_config.logs_dir,
            name=experiment_name
        )