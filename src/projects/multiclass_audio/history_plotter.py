import os
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, HTML

def _has(history: dict, key: str) -> bool:
    return key in history and bool(history[key])

def plot_history(history: dict, save_path: str | None = None, show_best: bool = True):
    """Сетка 1x2: Loss и Accuracy/F1 + опциональная табличка с лучшими метриками."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # [0] Loss
    if _has(history, "train_loss") or _has(history, "val_loss"):
        if _has(history, "train_loss"):
            axes[0].plot(history["train_loss"], label="Train")
        if _has(history, "val_loss"):
            axes[0].plot(history["val_loss"], label="Val")
        axes[0].set_title("Total Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
    else:
        axes[0].set_visible(False)

    # [1] Accuracy & F1
    drawn = False
    for key, label, style, color in [
        ("train_acc", "Train Acc", "-", "blue"),
        ("val_acc", "Val Acc", "--", "cornflowerblue"),
        ("train_f1", "Train F1", "-", "darkgreen"),
        ("val_f1", "Val F1", "--", "limegreen"),
    ]:
        if _has(history, key):
            axes[1].plot(history[key], label=label, linestyle=style, color=color)
            drawn = True

    if drawn:
        axes[1].set_title("Accuracy & F1")
        axes[1].set_xlabel("Epoch")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
    else:
        axes[1].set_visible(False)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    # --- Табличка с лучшими метриками ---
    if show_best and _has(history, "val_f1"):
        best_epoch = int(max(range(len(history["val_f1"])), key=lambda i: history["val_f1"][i])) + 1
        
        def _get(key):
            if key in history and len(history[key]) >= best_epoch:
                return f"{float(history[key][best_epoch - 1]):.4f}"
            return "—"

        # Собираем HTML-таблицу
        html = f"""
        <div style="margin-top: 10px;">
            <h4>🏆 Лучшая эпоха по Val F1: {best_epoch}</h4>
            <table style="border-collapse: collapse; font-size: 14px; width: 50%;">
                <tr style="background-color: #f2f2f2;">
                    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Метрика</th>
                    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Значение</th>
                </tr>
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px;">Val Loss</td>
                    <td style="border: 1px solid #ddd; padding: 8px;">{_get('val_loss')}</td>
                </tr>
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px;">Val Accuracy</td>
                    <td style="border: 1px solid #ddd; padding: 8px;">{_get('val_acc')}</td>
                </tr>
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px;">Val F1 (macro)</td>
                    <td style="border: 1px solid #ddd; padding: 8px;">{_get('val_f1')}</td>
                </tr>
            </table>
        </div>
        """
        display(HTML(html))


def plot_history_from_logs(log_dir="logs/multiclass_quickstart", save_path="reports/history.png", show_best=True):
    versions = sorted([d for d in os.listdir(log_dir) if d.startswith("version_")], key=lambda x: int(x.split("_")[1]))
    latest_csv = os.path.join(log_dir, versions[-1], "metrics.csv")
    
    df = pd.read_csv(latest_csv)
    history = {col: df[["epoch", col]].dropna().groupby("epoch")[col].last().tolist() 
               for col in df.columns if col not in ["epoch", "step"]}
    
    plot_history(history, save_path=save_path, show_best=show_best)