"""
Глобальные константы проекта.

Внимание: MEL_MEAN и MEL_STD - float, а не кортежи.
В старой версии стояли запятые в конце (MEL_MEAN=-41.9199,), что делало
их кортежами (-41.9199,) и ломало нормализацию.
"""
import audiomentations as A


class C:
    # ---- Аугментации для сырого аудио ----
    transform = A.Compose([
        A.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        A.PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
        A.TimeStretch(min_rate=0.8, max_rate=1.2, p=0.5),
        A.Shift(min_shift=-0.2, max_shift=0.2, shift_unit="fraction", p=0.5),
        A.Gain(min_gain_db=-10.0, max_gain_db=10.0, p=0.5),
    ])

    # ---- Воспроизводимость ----
    SEED = 42

    # ---- Нормализация мел-спектрограмм ----
    MEL_MEAN = -41.9199
    MEL_STD = 17.9253

    # ---- Данные ----
    BATCH_SIZE = 32
    DATA_SHAPE = (256, 173)          # (n_mels, time)
    NUM_CATEGORIES = 5
    NUM_TARGETS = 7

    # ---- Сплит ----
    TEST_SIZE = 0.15
    VAL_SIZE = 0.15

    # ---- Пути ----
    ENCODERS_DIR = "encoders"
    CHECKPOINTS_DIR = "checkpoints"
