"""
Аналитика по тестовой выборке для двухголовой аудио-модели.

Принимает сырые предсказания (dict[str, np.ndarray]) и умеет:
  - считать accuracy / F1 (weighted, macro, per-class)
  - считать combined accuracy и combined F1 (обе головы одновременно)
  - строить отдельные матрицы ошибок (cat, tgt)
  - строить комбинированную матрицу (cat+tgt)
  - возвращать DataFrame'ы для отчёта
  - мерить скорость инференса (если передан loader + model)

Ожидаемые ключи preds_dict:
    pred_cat, pred_tgt, true_cat, true_tgt  — 1D numpy-массивы индексов
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


class TestSummary:
    """
    Parameters
    ----------
    preds_dict : dict[str, np.ndarray]
        Ключи: pred_cat, pred_tgt, true_cat, true_tgt.
    category_mapping : dict[int, str] | None
        Индекс → имя категории.
    target_mapping : dict[int, str] | None
        Индекс → имя таргета.
    """

    def __init__(
        self,
        preds_dict: dict,
        category_mapping: Optional[dict] = None,
        target_mapping: Optional[dict] = None,
    ):
        # Приводим к numpy (на случай, если пришли torch.Tensor)
        self.cat_preds = self._to_numpy(preds_dict["pred_cat"])
        self.tgt_preds = self._to_numpy(preds_dict["pred_tgt"])
        self.cat_true = self._to_numpy(preds_dict["true_cat"])
        self.tgt_true = self._to_numpy(preds_dict["true_tgt"])

        self.category_mapping = category_mapping or {}
        self.target_mapping = target_mapping or {}

        self.metrics: dict = {}

    @staticmethod
    def _to_numpy(x):
        if hasattr(x, "cpu"):
            x = x.cpu().numpy()
        return np.asarray(x)

    # ------------------------------------------------------------------
    # Метрики
    # ------------------------------------------------------------------

    def compute_all(self) -> dict:
        """
        Считает все метрики и сохраняет в self.metrics.
        Возвращает self.metrics.
        """
        cat_preds, cat_true = self.cat_preds, self.cat_true
        tgt_preds, tgt_true = self.tgt_preds, self.tgt_true

        acc_cat = accuracy_score(cat_true, cat_preds)
        acc_tgt = accuracy_score(tgt_true, tgt_preds)

        f1_cat_weighted = f1_score(cat_true, cat_preds, average="weighted", zero_division=0)
        f1_tgt_weighted = f1_score(tgt_true, tgt_preds, average="weighted", zero_division=0)
        f1_cat_macro = f1_score(cat_true, cat_preds, average="macro", zero_division=0)
        f1_tgt_macro = f1_score(tgt_true, tgt_preds, average="macro", zero_division=0)

        f1_cat_per_class = f1_score(cat_true, cat_preds, average=None, zero_division=0)
        f1_tgt_per_class = f1_score(tgt_true, tgt_preds, average=None, zero_division=0)

        # Combined: обе головы угадали одновременно
        combined_correct = (cat_preds == cat_true) & (tgt_preds == tgt_true)
        combined_accuracy = float(np.mean(combined_correct))

        # Combined F1 (по составным меткам)
        combined_true = np.array([f"{c}_{t}" for c, t in zip(cat_true, tgt_true)])
        combined_preds = np.array([f"{c}_{t}" for c, t in zip(cat_preds, tgt_preds)])
        f1_combined_weighted = f1_score(combined_true, combined_preds, average="weighted", zero_division=0)
        f1_combined_macro = f1_score(combined_true, combined_preds, average="macro", zero_division=0)

        # Per-class accuracy
        cat_class_accuracy = self._per_class_accuracy(
            cat_true, cat_preds, self.category_mapping, "Category"
        )
        tgt_class_accuracy = self._per_class_accuracy(
            tgt_true, tgt_preds, self.target_mapping, "Target"
        )
        combined_per_class = self._combined_per_class_accuracy()

        # Имена классов для per-class F1
        cat_names = (
            [self.category_mapping.get(i, f"Cat_{i}") for i in sorted(set(cat_true) | set(cat_preds))]
            if self.category_mapping else [str(i) for i in range(len(f1_cat_per_class))]
        )
        tgt_names = (
            [self.target_mapping.get(i, f"Tgt_{i}") for i in sorted(set(tgt_true) | set(tgt_preds))]
            if self.target_mapping else [str(i) for i in range(len(f1_tgt_per_class))]
        )

        self.metrics = {
            # accuracy
            "acc_category": acc_cat,
            "acc_target": acc_tgt,
            "combined_accuracy": combined_accuracy,
            # f1
            "f1_category_weighted": f1_cat_weighted,
            "f1_target_weighted": f1_tgt_weighted,
            "f1_category_macro": f1_cat_macro,
            "f1_target_macro": f1_tgt_macro,
            "f1_combined_weighted": f1_combined_weighted,
            "f1_combined_macro": f1_combined_macro,
            # per-class
            "f1_category_per_class": dict(zip(cat_names[:len(f1_cat_per_class)], f1_cat_per_class)),
            "f1_target_per_class": dict(zip(tgt_names[:len(f1_tgt_per_class)], f1_tgt_per_class)),
            "category_accuracy_per_class": cat_class_accuracy,
            "target_accuracy_per_class": tgt_class_accuracy,
            "combined_accuracy_per_class": combined_per_class,
            # raw
            "cat_true": cat_true, "cat_preds": cat_preds,
            "tgt_true": tgt_true, "tgt_preds": tgt_preds,
            "category_names": cat_names, "target_names": tgt_names,
        }
        return self.metrics

    def _per_class_accuracy(self, true, preds, mapping, prefix):
        result = {}
        for cls in np.unique(true):
            mask = true == cls
            if mask.sum() > 0:
                name = mapping.get(int(cls), f"{prefix}_{cls}")
                result[name] = float((preds[mask] == cls).mean() * 100)
        return result

    def _combined_per_class_accuracy(self):
        result = {}
        for cat in np.unique(self.cat_true):
            for tgt in np.unique(self.tgt_true):
                mask = (self.cat_true == cat) & (self.tgt_true == tgt)
                if mask.sum() == 0:
                    continue
                correct = ((self.cat_preds[mask] == cat) & (self.tgt_preds[mask] == tgt)).sum()
                acc = float(correct / mask.sum() * 100)
                cat_name = self.category_mapping.get(int(cat), f"Cat_{cat}")
                tgt_name = self.target_mapping.get(int(tgt), f"Tgt_{tgt}")
                result[f"{cat_name} + {tgt_name}"] = acc
        return result

    # ------------------------------------------------------------------
    # DataFrame'ы
    # ------------------------------------------------------------------

    def metrics_to_dataframe(self) -> dict:
        """
        Возвращает DataFrame'ы:
          - "summary": строки = category/target/combined,
                       столбцы = accuracy, f1_macro, f1_weighted
          - "category": accuracy по категориям
          - "target":   accuracy по таргетам
          - "combined": combined accuracy по парам (category + target)
        """
        m = self.metrics
        if not m:
            raise RuntimeError("Сначала вызовите compute_all()")

        # ---- Сводка по задачам (широкий формат) ----
        summary = pd.DataFrame({
            "Task": ["category", "target", "combined"],
            "Accuracy": [
                m["acc_category"],
                m["acc_target"],
                m["combined_accuracy"],
            ],
            "F1 (macro)": [
                m["f1_category_macro"],
                m["f1_target_macro"],
                m["f1_combined_macro"],
            ],
            "F1 (weighted)": [
                m["f1_category_weighted"],
                m["f1_target_weighted"],
                m["f1_combined_weighted"],
            ],
        }).set_index("Task").round(4)

        # ---- Per-class accuracy ----
        cat_df = pd.DataFrame({
            "Category": list(m["category_accuracy_per_class"].keys()),
            "Accuracy (%)": list(m["category_accuracy_per_class"].values()),
        }).sort_values("Accuracy (%)", ascending=False).reset_index(drop=True)
        cat_df["Accuracy (%)"] = cat_df["Accuracy (%)"].round(2)

        tgt_df = pd.DataFrame({
            "Target": list(m["target_accuracy_per_class"].keys()),
            "Accuracy (%)": list(m["target_accuracy_per_class"].values()),
        }).sort_values("Accuracy (%)", ascending=False).reset_index(drop=True)
        tgt_df["Accuracy (%)"] = tgt_df["Accuracy (%)"].round(2)

        comb_df = pd.DataFrame({
            "Pair": list(m["combined_accuracy_per_class"].keys()),
            "Accuracy (%)": list(m["combined_accuracy_per_class"].values()),
        }).sort_values("Accuracy (%)", ascending=False).reset_index(drop=True)
        comb_df["Accuracy (%)"] = comb_df["Accuracy (%)"].round(2)

        return {
            "summary": summary,
            "category": cat_df,
            "target": tgt_df,
            "combined": comb_df,
        }

    # ------------------------------------------------------------------
    # Матрицы ошибок
    # ------------------------------------------------------------------

    def plot_separate_confusion_matrices(self, figsize=(15, 5), save_path=None):
        """Две отдельные матрицы: category и target."""
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        cm_cat = confusion_matrix(self.cat_true, self.cat_preds)
        cat_labels = [self.category_mapping.get(i, f"Cat_{i}") for i in range(cm_cat.shape[0])]
        sns.heatmap(cm_cat, annot=True, fmt="d", cmap="Blues",
                    xticklabels=cat_labels, yticklabels=cat_labels, ax=axes[0])
        axes[0].set_title("Confusion Matrix: Categories")
        axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")

        cm_tgt = confusion_matrix(self.tgt_true, self.tgt_preds)
        tgt_labels = [self.target_mapping.get(i, f"Tgt_{i}") for i in range(cm_tgt.shape[0])]
        sns.heatmap(cm_tgt, annot=True, fmt="d", cmap="Blues",
                    xticklabels=tgt_labels, yticklabels=tgt_labels, ax=axes[1])
        axes[1].set_title("Confusion Matrix: Targets")
        axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        return cm_cat, cm_tgt

    def plot_combined_confusion_matrix(self, figsize=(12, 10), save_path=None):
        """
        Комбинированная матрица (category + target).
        Пустые строки/столбцы фильтруются автоматически.
        """
        true_combined = [
            f"{self.category_mapping.get(int(c), f'Cat_{c}')}-{self.target_mapping.get(int(t), f'Tgt_{t}')}"
            for c, t in zip(self.cat_true, self.tgt_true)
        ]
        pred_combined = [
            f"{self.category_mapping.get(int(c), f'Cat_{c}')}-{self.target_mapping.get(int(t), f'Tgt_{t}')}"
            for c, t in zip(self.cat_preds, self.tgt_preds)
        ]

        unique_labels = sorted(set(true_combined + pred_combined))
        cm_full = confusion_matrix(true_combined, pred_combined, labels=unique_labels)

        # Фильтр пустых строк/столбцов
        non_empty_rows = np.any(cm_full != 0, axis=1)
        non_empty_cols = np.any(cm_full != 0, axis=0)
        keep = non_empty_rows & non_empty_cols
        cm_filtered = cm_full[keep][:, keep]
        filtered_labels = [label for i, label in enumerate(unique_labels) if keep[i]]

        plt.figure(figsize=figsize)
        cm_df = pd.DataFrame(cm_filtered, index=filtered_labels, columns=filtered_labels)
        sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", annot_kws={"size": 10})
        plt.title("Combined Confusion Matrix (Category + Target)", fontsize=14)
        plt.xlabel("Predicted", fontsize=12); plt.ylabel("True", fontsize=12)
        plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        return cm_filtered, filtered_labels

    # ------------------------------------------------------------------
    # Скорость инференса
    # ------------------------------------------------------------------

    @staticmethod
    def measure_inference_speed(model, loader, device="cuda") -> dict:
        """
        Замеряет скорость инференса.
        Возвращает {'avg_time_per_sample_ms', 'samples_per_second', 'total_samples'}.
        Не трогает self — можно вызывать как утилиту отдельно.
        """
        model.eval()
        total_samples = 0
        total_time = 0.0

        with torch.no_grad():
            for x, _, _ in loader:
                x = x.to(device)
                batch_size = x.size(0)
                total_samples += batch_size

                start = time.perf_counter()
                _ = model(x)
                if device == "cuda" and torch.cuda.is_available():
                    torch.cuda.synchronize()
                total_time += time.perf_counter() - start

        return {
            "avg_time_per_sample_ms": (total_time / total_samples) * 1000 if total_samples else 0.0,
            "samples_per_second": total_samples / total_time if total_time > 0 else 0.0,
            "total_samples": total_samples,
        }