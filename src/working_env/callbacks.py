import os
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    Callback,
)
from .config import env_config

class HistoryCollector(Callback):
    """Собирает метрики по эпохам для последующего построения графиков (plot_history)."""
    def __init__(self):
        super().__init__()
        self.history = {}

    def _append(self, key, value):
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, (int, float)):
            self.history.setdefault(key, []).append(float(value))

    def on_validation_epoch_end(self, trainer, pl_module):
        for key, value in trainer.callback_metrics.items():
            self._append(key, value)

    def get_history(self):
        return self.history


def get_default_callbacks(
    experiment_name: str = "default",
    monitor: str = "val_loss",
    mode: str = "min",
    patience: int = 10,
    save_top_k: int = 1,
):
    """Возвращает список стандартных коллбэков для обучения."""
    ckpt_dir = os.path.join(env_config.checkpoints_dir, experiment_name)

    return [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="best-{epoch:02d}-{" + monitor + ":.3f}",
            monitor=monitor,
            mode=mode,
            save_top_k=save_top_k,
            verbose=True,
        ),
        EarlyStopping(
            monitor=monitor,
            mode=mode,
            patience=patience,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        HistoryCollector(),
    ]