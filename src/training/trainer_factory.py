"""
Сборка Lightning Trainer с MLflow, чекпоинтами и early stopping.

Одна функция make_trainer(...) — и в ноутбуке не нужно помнить,
какие callbacks и какой logger подключать.
"""
from __future__ import annotations

from typing import Optional

import lightning as L
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)

from .callbacks import PredictionCollector


def make_trainer(
    experiment_name: str,
    run_name: str,
    *,
    max_epochs: int = 100,
    # ---- мониторинг ----
    monitor: str = "val_f1_mean",
    mode: str = "max",
    patience: int = 15,
    # ---- чекпоинты ----
    checkpoint_dir: Optional[str] = None,
    save_top_k: int = 1,
    # ---- MLflow ----
    tracking_uri: str = "file:./mlruns",
    log_model: bool = False,
    # ---- железо ----
    accelerator: str = "auto",
    devices: str | int = "auto",
    precision: str = "32-true",
    # ---- прочее ----
    gradient_clip_val: Optional[float] = 1.0,
    log_every_n_steps: int = 10,
    check_val_every_n_epoch: int = 1,
    with_prediction_collector: bool = False,
    collector_stage: str = "test",
    enable_progress_bar: bool = True,
):
    """
    Собирает Trainer + (опционально) PredictionCollector.

    Parameters
    ----------
    experiment_name, run_name : str
        Для MLflow. run_name — имя конкретного прогона.
    max_epochs : int
    monitor, mode : str
        Метрика и направление для ModelCheckpoint/EarlyStopping.
        По умолчанию val_f1_mean / max.
    patience : int
        Patience для EarlyStopping (в эпохах).
    checkpoint_dir : str | None
        Куда сохранять чекпоинты. None → ``checkpoints/{run_name}``.
    save_top_k : int
        Сколько лучших чекпоинтов хранить.
    tracking_uri : str
        MLflow tracking URI. ``file:./mlruns`` — локально, без сервера.
    log_model : bool
        Логировать ли чекпоинты как артефакты MLflow.
        В некоторых версиях MLflow (2.12–2.22) были баги с путями —
        если падает, поставьте False.
    accelerator, devices, precision
        Пробрасываются в Trainer. ``auto`` — сам выберет GPU/CPU.
    gradient_clip_val : float | None
        Обрезка градиентов. None — выключено.
    log_every_n_steps : int
        Как часто логировать шаговые метрики (у нас on_step=False, так что
        влияет в основном на прогресс-бар).
    check_val_every_n_epoch : int
        Как часто запускать валидацию.
    with_prediction_collector : bool
        Добавить ли PredictionCollector (нужен для TestSummary).
    collector_stage : str
        'test' или 'predict' — какую стадию собирать.
    enable_progress_bar : bool
        Показывать ли tqdm-бар.

    Returns
    -------
    trainer : L.Trainer
    collector : PredictionCollector | None
    """
    checkpoint_dir = checkpoint_dir or f"checkpoints/{run_name}"

    # ---- Logger ----
    mlf_logger = MLFlowLogger(
        experiment_name=experiment_name,
        run_name=run_name,
        tracking_uri=tracking_uri,
        log_model=log_model,
    )

    # ---- Callbacks ----
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
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
    ]

    collector = None
    if with_prediction_collector:
        collector = PredictionCollector(stage=collector_stage)
        callbacks.append(collector)

    # ---- Trainer ----
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        logger=mlf_logger,
        callbacks=callbacks,
        gradient_clip_val=gradient_clip_val,
        log_every_n_steps=log_every_n_steps,
        check_val_every_n_epoch=check_val_every_n_epoch,
        enable_progress_bar=enable_progress_bar,
    )

    return trainer, collector
