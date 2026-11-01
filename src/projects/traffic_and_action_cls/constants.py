import audiomentations as A

class C:
    SEED = 42
    
    # --- Нормализация мел-спектрограмм ---
    MEL_MEAN = -41.9199
    MEL_STD = 17.9253
    
    # --- Данные ---
    BATCH_SIZE = 32
    
    # --- Аугментации для сырого аудио ---
    transform = A.Compose([
        A.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        A.PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
        A.TimeStretch(min_rate=0.8, max_rate=1.2, p=0.5),
        A.Shift(min_shift=-0.2, max_shift=0.2, shift_unit="fraction", p=0.5),
        A.Gain(min_gain_db=-10.0, max_gain_db=10.0, p=0.5),
    ])

    # --- Имена классов (для графиков и TestSummary) ---
    # порядок индексов должен совпадать с тем, 
    # как они закодированы в вашем .npz файле (или в LabelEncoder)
    CATEGORY_NAMES = {
        0: "car",
        1: "emv",
        2: "motorcycle",
        3: "tram",
        4: "truck"
    }
    
    TARGET_NAMES = {
        0: "acceleration",
        1: "bell",
        2: "braking",
        3: "horn",
        4: "idling",
        5: "passing",
        6: "siren"
    }