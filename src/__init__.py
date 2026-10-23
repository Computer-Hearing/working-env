"""
Фреймворк для экспериментов с двухголовой аудио-моделью.

Основные точки входа:
    from src.constants import C
    from src.data import DataPipelineRaw, DataPipelineDone
    from src.models import MobileNetAudio, SmallAudioCNN, LitAudioModel
    from src.training import make_trainer
    from src.evaluation import TestSummary, plot_history
"""
__version__ = "0.1.0"
