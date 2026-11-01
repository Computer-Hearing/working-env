from __future__ import annotations

import os
from typing import Optional

import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder


DEFAULT_ENCODERS_DIR = "encoders"


def to_str_array(arr) -> np.ndarray:
    """
    Приводит массив меток к строкам.
    Нужно, чтобы типы были согласованы до fit/transform:
    LabelEncoder не любит смесь int и str.
    """
    return np.array([str(x) for x in arr], dtype=str)


def fit_or_load_encoders(
    train_category,
    train_target,
    category_path: Optional[str] = None,
    target_path: Optional[str] = None,
    encoders_dir: str = DEFAULT_ENCODERS_DIR,
    verbose: bool = True,
):
    """
    Возвращает обученные (или загруженные) энкодеры и словари для маппинга.

    Parameters
    ----------
    train_category, train_target : array-like
        Метки train (уже приведённые к str — см. to_str_array).
    category_path, target_path : str | None
        Пути к сохранённым .pkl. Если None или файлы не подходят —
        энкодеры обучаются заново и сохраняются в ``encoders_dir``.
    encoders_dir : str
        Куда сохранять, если путей нет или загрузка не удалась.
    verbose : bool
        Печатать ли прогресс.

    Returns
    -------
    enc_category : LabelEncoder
    enc_target : LabelEncoder
    category_to_idx : dict[str, int]
    target_to_idx : dict[str, int]
    idx_to_category : dict[int, str]
    idx_to_target : dict[int, str]
    """
    os.makedirs(encoders_dir, exist_ok=True)

    # Дефолтные пути, если не передали
    category_path = category_path or os.path.join(
        encoders_dir, "label_encoder_category.pkl"
    )
    target_path = target_path or os.path.join(
        encoders_dir, "label_encoder_target.pkl"
    )

    enc_category = _try_load(category_path, train_category, verbose)
    enc_target = _try_load(target_path, train_target, verbose)

    # Если хотя бы один не загрузился/не подошёл — обучаем оба заново,
    # чтобы словари были согласованы (иначе можно получить рассинхрон).
    if enc_category is None or enc_target is None:
        if verbose:
            print("Обучение энкодеров заново")
        enc_category = LabelEncoder().fit(train_category)
        enc_target = LabelEncoder().fit(train_target)

        # Сохраняем (перезаписываем)
        _save(enc_category, category_path, verbose)
        _save(enc_target, target_path, verbose)
    elif verbose:
        print("Энкодеры загружены")

    # Словари для маппинга (строки <-> индексы)
    category_to_idx = {
        str(cat): int(idx) for idx, cat in enumerate(enc_category.classes_)
    }
    target_to_idx = {
        str(tgt): int(idx) for idx, tgt in enumerate(enc_target.classes_)
    }
    idx_to_category = {int(idx): str(cat) for idx, cat in enumerate(enc_category.classes_)}
    idx_to_target = {int(idx): str(tgt) for idx, tgt in enumerate(enc_target.classes_)}

    return (
        enc_category,
        enc_target,
        category_to_idx,
        target_to_idx,
        idx_to_category,
        idx_to_target,
    )


# ----------------------------------------------------------------------
# Внутренние хелперы
# ----------------------------------------------------------------------

def _try_load(path: str, train_labels: np.ndarray, verbose: bool):
    """
    Пробует загрузить энкодер из .pkl.
    Возвращает None, если:
      - файла нет,
      - загрузка упала,
      - загруженный энкодер не знает все классы из train_labels.
    """
    if not os.path.exists(path):
        if verbose:
            print(f"  - {path} не найден, будет обучен заново")
        return None

    try:
        encoder = joblib.load(path)
    except Exception as e:
        if verbose:
            print(f"  - Не удалось загрузить {path}: {type(e).__name__}: {e}")
        return None

    # Проверка: энкодер должен знать все классы из train
    try:
        unseen = set(train_labels) - set(encoder.classes_)
    except AttributeError:
        # Файл загрузился, но это не LabelEncoder
        if verbose:
            print(f"  - {path} не похож на LabelEncoder, переобучаем")
        return None

    if unseen:
        if verbose:
            print(f"  - {path}: энкодер не знает классы {unseen}, переобучаем")
        return None

    return encoder


def _save(encoder, path: str, verbose: bool):
    """Сохраняет энкодер, создавая папку при необходимости."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(encoder, path)
    if verbose:
        print(f"  - Сохранён: {path}")