"""
Кастомные callbacks для Trainer.

PredictionCollector собирает выходы test_step/predict_step по батчам
и склеивает их в один словарь в on_test_end. Нужен для TestSummary:
Lightning сам по себе не агрегирует то, что test_step возвращает через return.
"""
from __future__ import annotations

from typing import Any

import torch
from lightning.pytorch.callbacks import Callback


class PredictionCollector(Callback):
    """
    Собирает словари, возвращённые из test_step/predict_step, в один.

    Использование:
        collector = PredictionCollector()
        trainer = Trainer(callbacks=[collector, ...])
        trainer.test(model, datamodule=dm)
        all_preds = collector.get_predictions()
        summary = TestSummary(all_preds)

    Ожидает, что test_step (или predict_step) возвращает dict[str, Tensor].
    """

    def __init__(self, stage: str = "test"):
        super().__init__()
        assert stage in ("test", "predict"), "stage должен быть 'test' или 'predict'"
        self.stage = stage
        self.batch_outputs: list[dict[str, torch.Tensor]] = []

    # ---- test ----
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if self.stage != "test":
            return
        if outputs is not None:
            self.batch_outputs.append(outputs)

    # ---- predict ----
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if self.stage != "predict":
            return
        if outputs is not None:
            self.batch_outputs.append(outputs)

    def on_test_start(self, trainer, pl_module):
        if self.stage == "test":
            self.batch_outputs.clear()

    def on_predict_start(self, trainer, pl_module):
        if self.stage == "predict":
            self.batch_outputs.clear()

    # ---- API для ноутбука ----
    def get_predictions(self) -> dict[str, torch.Tensor]:
        """
        Склеивает список батчей в единый словарь тензоров.
        Вызывать после trainer.test()/trainer.predict().
        """
        if not self.batch_outputs:
            raise RuntimeError(
                "PredictionCollector пуст. Убедитесь, что test_step/predict_step "
                "возвращает dict, и что trainer.test()/trainer.predict() уже выполнен."
            )
        keys = self.batch_outputs[0].keys()
        return {
            k: torch.cat([b[k].cpu() for b in self.batch_outputs], dim=0)
            for k in keys
        }


class HistoryCollector(Callback):
    """
    Накапливает метрики по эпохам в формате, совместимом с plot_history.

    Использование:
        collector = HistoryCollector()
        trainer = Trainer(callbacks=[collector, ...])
        trainer.fit(model, datamodule=dm)
        history = collector.get_history()
        plot_history(history, save_path="reports/history.png")

    Собирает всё, что залогировано через self.log(..., on_epoch=True),
    включая val_f1_mean, val_acc_both и lr (через LearningRateMonitor
    или из optimizer.param_groups).
    """

    def __init__(self):
        super().__init__()
        self.history: dict[str, list[float]] = {}

    def _append(self, key: str, value):
        self.history.setdefault(key, []).append(float(value))

    def on_train_epoch_end(self, trainer, pl_module):
        # Метрики train-эпохи
        metrics = trainer.callback_metrics
        for key in (
            "train_loss", "train_loss_cat", "train_loss_tgt",
            "train_acc_cat", "train_acc_tgt",
        ):
            if key in metrics:
                self._append(key, metrics[key])

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        for key in (
            "val_loss", "val_loss_cat", "val_loss_tgt",
            "val_acc_cat", "val_acc_tgt",
            "val_f1_cat", "val_f1_tgt", "val_f1_mean",
            "val_acc_both",
        ):
            if key in metrics:
                self._append(key, metrics[key])

        # Learning rate из оптимизатора
        optimizers = trainer.optimizers
        if optimizers:
            self._append("lr", optimizers[0].param_groups[0]["lr"])

    def on_test_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        for key in (
            "test_loss", "test_loss_cat", "test_loss_tgt",
            "test_acc_cat", "test_acc_tgt",
            "test_f1_cat", "test_f1_tgt", "test_f1_mean",
            "test_acc_both",
        ):
            if key in metrics:
                self._append(key, metrics[key])

    def on_fit_start(self, trainer, pl_module):
        self.history.clear()

    def get_history(self) -> dict[str, list[float]]:
        """Возвращает накопленную историю (можно сразу в plot_history)."""
        return self.history
