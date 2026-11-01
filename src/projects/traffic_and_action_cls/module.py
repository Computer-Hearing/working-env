from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics import Accuracy, F1Score


class LitTrafficModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        num_categories: int = 5,
        num_targets: int = 7,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        scheduler_factor: float = 0.5,
        scheduler_patience: int = 5,
        criterion_cat: nn.Module | None = None,
        criterion_tgt: nn.Module | None = None,
        loss_weights: tuple[float, float] = (1.0, 1.0)
    ):
        super().__init__()
        # save_hyperparameters логирует всё, кроме самой модели (её нельзя pickle-нуть в hparams)
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.num_categories = num_categories
        self.num_targets = num_targets
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience

        # ---- Метрики (torchmetrics сами агрегируют по эпохе) ----
        # train
        self.train_acc_cat = Accuracy(task="multiclass", num_classes=num_categories)
        self.train_acc_tgt = Accuracy(task="multiclass", num_classes=num_targets)

        # val
        self.val_acc_cat = Accuracy(task="multiclass", num_classes=num_categories)
        self.val_acc_tgt = Accuracy(task="multiclass", num_classes=num_targets)
        self.val_f1_cat = F1Score(task="multiclass", num_classes=num_categories, average="macro")
        self.val_f1_tgt = F1Score(task="multiclass", num_classes=num_targets, average="macro")

        # test (отдельные объекты, чтобы не смешивать с val)
        self.test_acc_cat = Accuracy(task="multiclass", num_classes=num_categories)
        self.test_acc_tgt = Accuracy(task="multiclass", num_classes=num_targets)
        self.test_f1_cat = F1Score(task="multiclass", num_classes=num_categories, average="macro")
        self.test_f1_tgt = F1Score(task="multiclass", num_classes=num_targets, average="macro")
        
        # Если loss не передан, используем дефолтный CrossEntropy
        self.criterion_cat = criterion_cat if criterion_cat is not None else nn.CrossEntropyLoss()
        self.criterion_tgt = criterion_tgt if criterion_tgt is not None else nn.CrossEntropyLoss()
        self.loss_weights = loss_weights

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x):
        return self.model(x)

    # ------------------------------------------------------------------
    # Общий шаг
    # ------------------------------------------------------------------

    def _shared_step(self, batch, stage: str):
        x, y_cat_onehot, y_tgt_onehot = batch
        y_cat = torch.argmax(y_cat_onehot, dim=1)
        y_tgt = torch.argmax(y_tgt_onehot, dim=1)

        logits_cat, logits_tgt = self(x)

        loss_cat = self.criterion_cat(logits_cat, y_cat)
        loss_tgt = self.criterion_tgt(logits_tgt, y_tgt)
        
        w_cat, w_tgt = self.loss_weights
        loss = w_cat * loss_cat + w_tgt * loss_tgt

        # ---- Логирование loss ----
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f"{stage}_loss_cat", loss_cat, on_epoch=True, on_step=False)
        self.log(f"{stage}_loss_tgt", loss_tgt, on_epoch=True, on_step=False)

        preds_cat = torch.argmax(logits_cat, dim=1)
        preds_tgt = torch.argmax(logits_tgt, dim=1)

        # ---- Метрики по стадиям ----
        if stage == "train":
            self.train_acc_cat(preds_cat, y_cat)
            self.train_acc_tgt(preds_tgt, y_tgt)
            self.log("train_acc_cat", self.train_acc_cat, on_epoch=True, on_step=False)
            self.log("train_acc_tgt", self.train_acc_tgt, on_epoch=True, on_step=False)

        elif stage == "val":
            self.val_acc_cat(preds_cat, y_cat)
            self.val_acc_tgt(preds_tgt, y_tgt)
            self.val_f1_cat(preds_cat, y_cat)
            self.val_f1_tgt(preds_tgt, y_tgt)

            self.log("val_acc_cat", self.val_acc_cat, on_epoch=True, on_step=False)
            self.log("val_acc_tgt", self.val_acc_tgt, on_epoch=True, on_step=False)
            self.log("val_f1_cat", self.val_f1_cat, on_epoch=True, on_step=False)
            self.log("val_f1_tgt", self.val_f1_tgt, on_epoch=True, on_step=False)

            # Комбинированная метрика: обе головы угадали одновременно.
            # Считаем батч-локально и усредняем по эпохе.
            both_correct = ((preds_cat == y_cat) & (preds_tgt == y_tgt)).float().mean()
            self.log("val_acc_both", both_correct, on_epoch=True, on_step=False)

            self.log("val_f1_cat", self.val_f1_cat, on_epoch=True, on_step=False)
            self.log("val_f1_tgt", self.val_f1_tgt, on_epoch=True, on_step=False)

        elif stage == "test":
            self.test_acc_cat(preds_cat, y_cat)
            self.test_acc_tgt(preds_tgt, y_tgt)
            self.test_f1_cat(preds_cat, y_cat)
            self.test_f1_tgt(preds_tgt, y_tgt)

            self.log("test_acc_cat", self.test_acc_cat, on_epoch=True, on_step=False)
            self.log("test_acc_tgt", self.test_acc_tgt, on_epoch=True, on_step=False)
            self.log("test_f1_cat", self.test_f1_cat, on_epoch=True, on_step=False)
            self.log("test_f1_tgt", self.test_f1_tgt, on_epoch=True, on_step=False)

            both_correct = ((preds_cat == y_cat) & (preds_tgt == y_tgt)).float().mean()
            self.log("test_acc_both", both_correct, on_epoch=True, on_step=False)

        return loss

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------
    def on_validation_epoch_end(self):
        """
        Средний F1 по двум головам. Считается после того, как метрики
        накопили состояние по всей эпохе валидации.
        """
        mean_f1 = (self.val_f1_cat.compute() + self.val_f1_tgt.compute()) / 2
        self.log("val_f1_mean", mean_f1, on_epoch=True, on_step=False, prog_bar=True)

    def on_test_epoch_end(self):
        """Средний F1 по двум головам на тесте."""
        mean_f1 = (self.test_f1_cat.compute() + self.test_f1_tgt.compute()) / 2
        self.log("test_f1_mean", mean_f1, on_epoch=True, on_step=False)

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Возвращает сырые предсказания для TestSummary.
        Не логирует метрики — только отдаёт данные наружу.
        """
        x, y_cat_onehot, y_tgt_onehot = batch
        logits_cat, logits_tgt = self(x)

        return {
            "pred_cat": torch.argmax(logits_cat, dim=1),
            "pred_tgt": torch.argmax(logits_tgt, dim=1),
            "true_cat": torch.argmax(y_cat_onehot, dim=1),
            "true_tgt": torch.argmax(y_tgt_onehot, dim=1),
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.scheduler_factor,
            patience=self.scheduler_patience,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }