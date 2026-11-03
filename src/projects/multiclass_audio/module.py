import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, F1Score
from typing import Optional, Callable
import random
import numpy as np


class LitMulticlassModule(L.LightningModule):
    def __init__(
        self, 
        model: nn.Module, 
        num_classes: int, 
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        criterion: Optional[Callable] = None,
        optimizer_cls: type = torch.optim.Adam,
        optimizer_kwargs: Optional[dict] = None,
        scheduler_cls: Optional[type] = None,
        scheduler_kwargs: Optional[dict] = None,
        scheduler_monitor: str = "val_loss",
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "criterion", "optimizer_cls", "scheduler_cls"])
        
        self.model = model
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.scheduler_cls = scheduler_cls
        self.scheduler_kwargs = scheduler_kwargs or {}
        self.scheduler_monitor = scheduler_monitor
        
        self.seed = seed
        if seed is not None:
            self._set_seed(seed)

        # Train
        self.train_acc_metric = Accuracy(task="multiclass", num_classes=num_classes)
        self.train_f1_metric = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        
        # Val
        self.val_acc_metric = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1_metric = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        
        # Test
        self.test_acc_metric = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_f1_metric = F1Score(task="multiclass", num_classes=num_classes, average="macro")

    @staticmethod
    def _set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        L.seed_everything(seed, workers=True)

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, stage):
        x, y = batch
        logits = self(x)
        
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)

        # Логируем loss на каждом этапе
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=False)

        # Логируем метрики
        if stage == "train":
            self.train_acc_metric(preds, y)
            self.train_f1_metric(preds, y)
            self.log("train_acc", self.train_acc_metric, on_epoch=True, on_step=False)
            self.log("train_f1", self.train_f1_metric, on_epoch=True, on_step=False)
            
        elif stage == "val":
            self.val_acc_metric(preds, y)
            self.val_f1_metric(preds, y)
            self.log("val_acc", self.val_acc_metric, on_epoch=True, on_step=False)
            self.log("val_f1", self.val_f1_metric, on_epoch=True, on_step=False, prog_bar=True)
            
        elif stage == "test":
            self.test_acc_metric(preds, y)
            self.test_f1_metric(preds, y)
            self.log("test_acc", self.test_acc_metric, on_epoch=True, on_step=False)
            self.log("test_f1", self.test_f1_metric, on_epoch=True, on_step=False)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def predict_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        return {"pred": torch.argmax(logits, dim=1), "true": y}

    def configure_optimizers(self):
        opt_kwargs = {"lr": self.lr, **self.optimizer_kwargs}
        if self.weight_decay > 0:
            opt_kwargs["weight_decay"] = self.weight_decay
            
        optimizer = self.optimizer_cls(self.parameters(), **opt_kwargs)
        
        if self.scheduler_cls is None:
            return optimizer
            
        scheduler = self.scheduler_cls(optimizer, **self.scheduler_kwargs)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.scheduler_monitor,
                "interval": "epoch",
                "frequency": 1,
            }
        }