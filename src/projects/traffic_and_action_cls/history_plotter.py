import os
import numpy as np
import matplotlib.pyplot as plt

def _has(history: dict, key: str) -> bool:
    """Проверяет, есть ли ключ и непустой ли список."""
    return key in history and bool(history[key])

def _safe_legend(ax, *args, **kwargs):
    """legend() только если есть artists с label."""
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(*args, **kwargs)

def plot_history(history: dict, save_path: str | None = None):
    """
    Строит сетку 2x3:
        [0,0] Loss
        [0,1] Accuracy (Cat vs Tgt)
        [0,2] F1 (Cat vs Tgt)
        [1,0] Combined Accuracy (Cat & Tgt)
        [1,1] Learning Rate
        [1,2] Сводная таблица лучшей эпохи
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # --------------------------------------------------------------
    # [0,0] Loss
    # --------------------------------------------------------------
    if _has(history, "train_loss") or _has(history, "val_loss"):
        if _has(history, "train_loss"):
            axes[0, 0].plot(history["train_loss"], label="Train")
        if _has(history, "val_loss"):
            axes[0, 0].plot(history["val_loss"], label="Val")
        axes[0, 0].set_title("Total Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].grid(True, alpha=0.3)
        _safe_legend(axes[0, 0])
    else:
        axes[0, 0].set_visible(False)

    # --------------------------------------------------------------
    # [0,1] Accuracy (Cat vs Tgt)
    # --------------------------------------------------------------
    acc_lines = [
        ("train_acc_cat", "Train Cat", "-"),
        ("train_acc_tgt", "Train Tgt", "-"),
        ("val_acc_cat", "Val Cat", "--"),
        ("val_acc_tgt", "Val Tgt", "--"),
    ]
    drawn = False
    for key, label, style in acc_lines:
        if _has(history, key):
            axes[0, 1].plot(history[key], label=label, linestyle=style)
            drawn = True
    if drawn:
        axes[0, 1].set_title("Accuracy (Cat vs Tgt)")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].grid(True, alpha=0.3)
        _safe_legend(axes[0, 1])
    else:
        axes[0, 1].set_visible(False)

    # --------------------------------------------------------------
    # [0,2] F1 (Cat vs Tgt)
    # --------------------------------------------------------------
    f1_lines = [
        ("train_f1_cat", "Train Cat", "-"),
        ("train_f1_tgt", "Train Tgt", "-"),
        ("val_f1_cat", "Val Cat", "--"),
        ("val_f1_tgt", "Val Tgt", "--"),
    ]
    drawn = False
    for key, label, style in f1_lines:
        if _has(history, key):
            axes[0, 2].plot(history[key], label=label, linestyle=style)
            drawn = True
    if drawn:
        axes[0, 2].set_title("F1 Score (Cat vs Tgt)")
        axes[0, 2].set_xlabel("Epoch")
        axes[0, 2].grid(True, alpha=0.3)
        _safe_legend(axes[0, 2])
    else:
        axes[0, 2].set_visible(False)

    # --------------------------------------------------------------
    # [1,0] Combined Accuracy
    # --------------------------------------------------------------
    if _has(history, "val_acc_both"):
        if _has(history, "train_acc_both"):
            axes[1, 0].plot(history["train_acc_both"], label="Train")
        axes[1, 0].plot(history["val_acc_both"], label="Val")

        best = max(history["val_acc_both"])
        best_epoch = int(np.argmax(history["val_acc_both"])) + 1
        axes[1, 0].axhline(
            y=best, color="g", linestyle="--", alpha=0.5,
            label=f"Best {best:.4f} (ep {best_epoch})",
        )

        axes[1, 0].set_title("Combined Accuracy (Cat & Tgt)")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].grid(True, alpha=0.3)
        _safe_legend(axes[1, 0])
    else:
        axes[1, 0].set_visible(False)

    # --------------------------------------------------------------
    # [1,1] Learning Rate
    # --------------------------------------------------------------
    if _has(history, "lr"):
        axes[1, 1].plot(history["lr"])
        axes[1, 1].set_title("Learning Rate")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].set_visible(False)

    # --------------------------------------------------------------
    # [1,2] Сводная таблица
    # --------------------------------------------------------------
    axes[1, 2].axis("off")
    if _has(history, "val_acc_both"):
        best_epoch = int(np.argmax(history["val_acc_both"])) + 1

        def _get(key: str) -> float:
            if key in history and len(history[key]) >= best_epoch:
                return float(history[key][best_epoch - 1])
            return float("nan")

        text = (
            f"Best Results (Epoch {best_epoch}):\n"
            f"├─ Val Acc (Cat):      {_get('val_acc_cat'):.4f}\n"
            f"├─ Val Acc (Tgt):      {_get('val_acc_tgt'):.4f}\n"
            f"├─ Val Acc (Combined): {_get('val_acc_both'):.4f}\n"
            f"├─ Val F1 (Cat):       {_get('val_f1_cat'):.4f}\n"
            f"├─ Val F1 (Tgt):       {_get('val_f1_tgt'):.4f}\n"
            f"└─ Val F1 (Mean):      {_get('val_f1_mean'):.4f}"
        )
        axes[1, 2].text(
            0.1, 0.5, text, fontsize=11, verticalalignment="center",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
        )

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()