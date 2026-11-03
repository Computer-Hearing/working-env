import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report


class TestSummaryMulticlass:
    def __init__(self, preds_dict: dict, class_mapping: dict = None):
        # Приводим к numpy (на случай, если пришли тензоры)
        self.preds = self._to_numpy(preds_dict["pred"])
        self.true = self._to_numpy(preds_dict["true"])
        self.class_mapping = class_mapping or {}
        
        # Метрики будут здесь
        self.metrics = {}

    @staticmethod
    def _to_numpy(x):
        if hasattr(x, "cpu"):
            x = x.cpu().numpy()
        return np.asarray(x)

    def compute_all(self):
        acc = accuracy_score(self.true, self.preds)
        f1_macro = f1_score(self.true, self.preds, average="macro", zero_division=0)
        f1_weighted = f1_score(self.true, self.preds, average="weighted", zero_division=0)
        
        self.metrics = {
            "acc": acc,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
        }
        return self.metrics

    def print_report(self):
        if not self.metrics:
            self.compute_all()
        
        m = self.metrics
        print(f"Accuracy:    {m['acc']:.4f}")
        print(f"F1 (macro):  {m['f1_macro']:.4f}")
        print(f"F1 (weighted): {m['f1_weighted']:.4f}\n")
        
        # Classification report
        names = [self.class_mapping.get(i, str(i)) for i in sorted(set(self.true) | set(self.preds))]
        print(classification_report(self.true, self.preds, target_names=names, digits=4))

    def plot_confusion_matrix(self, save_path=None):
        cm = confusion_matrix(self.true, self.preds)
        names = [self.class_mapping.get(i, str(i)) for i in range(cm.shape[0])]
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=names, yticklabels=names)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            plt.savefig(save_path, dpi=150)
        plt.show()

    @staticmethod
    def measure_inference_speed(model, loader, device="cuda") -> dict:
        """Замеряет скорость инференса (мс на сэмпл и сэмплов в секунду)."""
        model.eval()
        total_samples = 0
        total_time = 0.0

        with torch.no_grad():
            for batch in loader:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
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