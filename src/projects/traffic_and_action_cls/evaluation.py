from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report


class TestSummary:
    def __init__(
        self,
        preds_dict: dict,
        category_mapping: Optional[dict] = None,
        target_mapping: Optional[dict] = None,
    ):
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
        cat_preds, cat_true = self.cat_preds, self.cat_true
        tgt_preds, tgt_true = self.tgt_preds, self.tgt_true

        acc_cat = accuracy_score(cat_true, cat_preds)
        acc_tgt = accuracy_score(tgt_true, tgt_preds)

        f1_cat_weighted = f1_score(cat_true, cat_preds, average="weighted", zero_division=0)
        f1_tgt_weighted = f1_score(tgt_true, tgt_preds, average="weighted", zero_division=0)
        f1_cat_macro = f1_score(cat_true, cat_preds, average="macro", zero_division=0)
        f1_tgt_macro = f1_score(tgt_true, tgt_preds, average="macro", zero_division=0)

        combined_correct = (cat_preds == cat_true) & (tgt_preds == tgt_true)
        combined_accuracy = float(np.mean(combined_correct))

        combined_true = np.array([f"{c}_{t}" for c, t in zip(cat_true, tgt_true)])
        combined_preds = np.array([f"{c}_{t}" for c, t in zip(cat_preds, tgt_preds)])
        f1_combined_weighted = f1_score(combined_true, combined_preds, average="weighted", zero_division=0)
        f1_combined_macro = f1_score(combined_true, combined_preds, average="macro", zero_division=0)

        self.metrics = {
            "acc_category": acc_cat, "acc_target": acc_tgt, "combined_accuracy": combined_accuracy,
            "f1_category_weighted": f1_cat_weighted, "f1_target_weighted": f1_tgt_weighted,
            "f1_category_macro": f1_cat_macro, "f1_target_macro": f1_tgt_macro,
            "f1_combined_weighted": f1_combined_weighted, "f1_combined_macro": f1_combined_macro,
        }
        return self.metrics

    def print_report(self):
        """Выводит полный текстовый отчет (сводка + classification_report)."""
        if not self.metrics:
            self.compute_all()
            
        m = self.metrics
        print("="*60)
        print("ОБЩАЯ СВОДКА")
        print("="*60)
        print(f"Category Accuracy:   {m['acc_category']:.4f}")
        print(f"Category F1 (macro): {m['f1_category_macro']:.4f}")
        print(f"Target Accuracy:     {m['acc_target']:.4f}")
        print(f"Target F1 (macro):   {m['f1_target_macro']:.4f}")
        print(f"Combined Accuracy:   {m['combined_accuracy']:.4f}  <-- обе головы угадали")
        print(f"Combined F1 (macro): {m['f1_combined_macro']:.4f} <-- F1 по совместным парам")
        
        cat_names = [self.category_mapping.get(i, str(i)) for i in sorted(set(self.cat_true) | set(self.cat_preds))]
        tgt_names = [self.target_mapping.get(i, str(i)) for i in sorted(set(self.tgt_true) | set(self.tgt_preds))]

        print("\n" + "="*60)
        print("CLASSIFICATION REPORT: CATEGORY")
        print("="*60)
        print(classification_report(self.cat_true, self.cat_preds, target_names=cat_names, digits=4))

        print("\n" + "="*60)
        print("CLASSIFICATION REPORT: TARGET")
        print("="*60)
        print(classification_report(self.tgt_true, self.tgt_preds, target_names=tgt_names, digits=4))

    def metrics_to_dataframe(self) -> dict:
        """Возвращает DataFrame'ы для визуализации."""
        m = self.metrics
        if not m:
            raise RuntimeError("Сначала вызовите compute_all()")

        summary = pd.DataFrame({
            "Task": ["category", "target", "combined"],
            "Accuracy": [m["acc_category"], m["acc_target"], m["combined_accuracy"]],
            "F1 macro": [m["f1_category_macro"], m["f1_target_macro"], m["f1_combined_macro"]],
            "F1 weighted": [m["f1_category_weighted"], m["f1_target_weighted"], m["f1_combined_weighted"]],
        }).set_index("Task").round(4)

        cat_accs = {self.category_mapping.get(c, f"Cat_{c}"): (self.cat_preds[self.cat_true == c] == c).mean() * 100 for c in np.unique(self.cat_true)}
        cat_df = pd.DataFrame(list(cat_accs.items()), columns=["Category", "Accuracy (%)"]).sort_values("Accuracy (%)", ascending=False).round(2)

        tgt_accs = {self.target_mapping.get(t, f"Tgt_{t}"): (self.tgt_preds[self.tgt_true == t] == t).mean() * 100 for t in np.unique(self.tgt_true)}
        tgt_df = pd.DataFrame(list(tgt_accs.items()), columns=["Target", "Accuracy (%)"]).sort_values("Accuracy (%)", ascending=False).round(2)

        comb_accs = {}
        for c in np.unique(self.cat_true):
            for t in np.unique(self.tgt_true):
                mask = (self.cat_true == c) & (self.tgt_true == t)
                if mask.sum() > 0:
                    acc = ((self.cat_preds[mask] == c) & (self.tgt_preds[mask] == t)).mean() * 100
                    c_name = self.category_mapping.get(c, f"Cat_{c}")
                    t_name = self.target_mapping.get(t, f"Tgt_{t}")
                    comb_accs[f"{c_name} + {t_name}"] = acc
        comb_df = pd.DataFrame(list(comb_accs.items()), columns=["Pair", "Accuracy (%)"]).sort_values("Accuracy (%)", ascending=False).round(2)

        return {"summary": summary, "category": cat_df, "target": tgt_df, "combined": comb_df}

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
        """Комбинированная матрица (category + target)."""
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
        """Замеряет скорость инференса."""
        model.eval()
        total_samples = 0
        total_time = 0.0

        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch

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