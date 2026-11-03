"""
Константы для проекта multiclass_audio.
"""

class C:
    SEED = 42
    BATCH_SIZE = 32
    
    # --- Нормализация (для датасетов road_surface) ---
    MEL_MEAN = 0.0
    MEL_STD = 1.0
    
    # --- Параметры спектрограмм ---
    DATA_SHAPE = (256, 173)
    
    # --- SpecAugment ---
    TIME_MASK_MAX = 20
    FREQ_MASK_MAX = 20
    N_MASKS = 2
    GAIN_RANGE = (0.8, 1.2)
    NOISE_SIGMA = 0.0
    
    # --- Имена классов для road_surface ---
    CLASS_NAMES = {
        0: "asphalt",
        1: "gravel",
        2: "other",
    }