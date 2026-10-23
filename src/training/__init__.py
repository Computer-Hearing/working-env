"""
Обучение: сборка Trainer с MLflow и callbacks.
"""
from .trainer_factory import make_trainer
from .callbacks import PredictionCollector, HistoryCollector

__all__ = ["make_trainer", "PredictionCollector", "HistoryCollector"]
