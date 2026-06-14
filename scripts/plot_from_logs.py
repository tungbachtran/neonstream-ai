"""
scripts/plot_from_logs.py
Vẽ biểu đồ trực tiếp từ data parse ra log — không cần JSON.
Data lấy từ training run: 2026-06-14 (16 epochs, CPU, 4 nhãn)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

OUTPUT_DIR = Path("outputs/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_NAMES = ["Clean", "Toxic", "Spam", "Adult"]
COLORS      = ["#4CAF50", "#F44336", "#FF9800", "#9C27B0"]

# ════════════════════════════════════════════════════════════
# DATA PARSE TỪ LOG (training run 2026-06-14, 16 epochs)
# ════════════════════════════════════════════════════════════
epochs = list(range(1, 17))

train_loss = [
    1.1395, 0.6774, 0.6096, 0.5457, 0.4996,
    0.4721, 0.4530, 0.4460, 0.4428, 0.4412,
    0.4407, 0.4390, 0.4385, 0.4377, 0.4385, 0.4368
]

val_loss = [
    0.8940, 0.9370, 1.0661, 0.7756, 0.6384,
    0.6240, 0.6769, 0.7275, 0.6999, 0.7852,
    0.6876, 0.8600, 0.7765, 0.7188, 0.6913, 0.7544
]

val_f1 = [
    0.3730, 0.4837, 0.5086, 0.5728, 0.7267,
    0.7248, 0.7029, 0.6773, 0.7914, 0.7263,
    0.8251, 0.6691, 0.7521, 0.7075, 0.7377, 0.7071
]

val_accuracy = [
    0.7818, 0.7773, 0.7489, 0.8797, 0.9342,
    0.9215, 0.9170, 0.8946, 0.9118, 0.8812,
    0.9178, 0.8505, 0.8901, 0.9081, 0.9141, 0.8939
]

# Learning rate per epoch (từ log)
learning_rates = [
    5.00e-06, 1.00e-05, 1.50e-05, 2.00e-05, 2.00e-05,
    1.98e-05, 1.97e-05, 1.94e-05, 1.91e-05, 1.87e-05,
    1.82e-05, 1.77e-05, 1.71e-05, 1.64e-05, 1.57e-05, 1.50e-05
]

# Confusion matrix từ TEST SET (epoch 11 - best)
# [[1172   77    0    3]
#  [  20   60    0    0]
#  [   0    0    3    0]
#  [   0    0    0    3]]
cm = np.array([
    [1172, 77,  0, 3],
    [  20, 60,  0, 0],
    [   0,  0,  3, 0],
    [   0,  0,  0, 3]
])

# F1 per class từ TEST SET (epoch 11)
final_f1_per_class = {
    "Clean": 0.9591,
    "Toxic": 0.5530,
    "Spam":  1.0000,
    "Adult": 0.6667
}

# Precision / Recall per class từ TEST SET
precision_per_class = [0.9832, 0.4380, 1.0000, 0.5000]
recall_per_class    = [0.9361, 0.7500, 1.0000, 1.0000]
f1_per_class_list   = [0.9591, 0.5530, 1.0000, 0.6667]

# Support (số mẫu thật) từ TEST SET
support = [1252, 80, 3, 3]

# Early stopping info
best_epoch    = 11
best_f1       = 0.8251
stopped_epoch = 16

print("Data loaded. Bat dau ve bieu do...")
print(f"Best epoch: {best_epoch} | Best F1: {best_f1} | Stopped: {stopped_epoch}")
print("-" * 60)

# ════════════════════════════════════════════════════════════
# BIEU DO 1: Train Loss vs Val Loss
# ════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(epochs, train_loss, "b-o", label="Train Loss", linewidth=2, markersize=5)
ax.plot(epochs, val_loss,   "r-s", label="Val Loss",   linewidth=2, markersize=5)

# Best epoch
ax.axvline(best_epoch, color="green", linestyle="--", linewidth=1.5,
           label=f"Best epoch {best_epoch} (F1={best_f1:.4f})")
ax.axvline(stopped_epoch, color="gray", linestyle=":", linewidth=1.5,
           label=f"Early stop ep{stopped_epoch}")

# Annotate min val loss
min_val_idx = int(np.argmin(val_loss))
ax.annotate(f"min={min(val_loss):.4f}",
            xy=(epochs[min_val_idx], min(val_loss)),
            xytext=(epochs[min_val_idx]+0.5, min(val_loss)+0.05),
            arrowprops=dict(arrowstyle="->", color="red"),
            color="red", fontsize=9)

ax.set_title("Train Loss vs Validation Loss (16 Epochs)", fontsize=14, fontweight="bold")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
ax.set_xticks(epochs)
ax.legend(loc="upper right"); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_loss_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 01_loss_curve.png")

# ════════════════════════════════════════════════════════════
# BIEU DO 2: Validation Accuracy
# ════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(epochs, val_accuracy, "r-s", label="Val Accuracy", linewidth=2, markersize=5)
ax.fill_between(epochs, val_accuracy, alpha=0.1, color="red")

best_acc_ep  = int(np.argmax(val_accuracy)) + 1
best_acc_val = max(val_accuracy)
ax.axvline(best_acc_ep, color="green", linestyle="--", linewidth=1.5,
           label=f"Best acc ep{best_acc_ep} ({best_acc_val:.4f})")

# Annotate mỗi điểm
for i, (ep, acc) in enumerate(zip(epochs, val_accuracy)):
    ax.annotate(f"{acc:.3f}", (ep, acc),
                textcoords="offset points", xytext=(0, 7),
                ha="center", fontsize=7, color="darkred")

ax.set_title("Validation Accuracy per Epoch", fontsize=14, fontweight="bold")
ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
ax.set_xticks(epochs); ax.set_ylim(0.7, 1.02)
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_accuracy_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 02_accuracy_curve.png")

# ════════════════════════════════════════════════════════════
# BIEU DO 3: F1 Macro per Epoch
# ════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(epochs, val_f1, "g-^", linewidth=2, markersize=6, label="Val F1 Macro")
ax.fill_between(epochs, val_f1, alpha=0.12, color="green")

ax.axvline(best_epoch, color="purple", linestyle="--", linewidth=1.5,
           label=f"Best F1 ep{best_epoch} ({best_f1:.4f})")

for i, (ep, f1) in enumerate(zip(epochs, val_f1)):
    ax.annotate(f"{f1:.3f}", (ep, f1),
                textcoords="offset points", xytext=(0, 8),
                ha="center", fontsize=7, color="darkgreen")

ax.set_title("Validation F1-Score (Macro) per Epoch", fontsize=14, fontweight="bold")
ax.set_xlabel("Epoch"); ax.set_ylabel("F1 Score")
ax.set_xticks(epochs); ax.set_ylim(0.2, 1.0)
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_f1_macro_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 03_f1_macro_curve.png")

# ════════════════════════════════════════════════════════════
# BIEU DO 4: Loss + F1 Dual Axis
# ════════════════════════════════════════════════════════════
fig, ax1 = plt.subplots(figsize=(12, 5))
ax2 = ax1.twinx()

l1, = ax1.plot(epochs, val_loss, "r-o", linewidth=2, markersize=4, label="Val Loss")
l2, = ax2.plot(epochs, val_f1,   "g-s", linewidth=2, markersize=4, label="Val F1 Macro")
l3, = ax1.plot(epochs, train_loss, "b--o", linewidth=1.5, markersize=3,
               alpha=0.6, label="Train Loss")

ax1.axvline(best_epoch, color="purple", linestyle="--", alpha=0.7,
            label=f"Best ep{best_epoch}")

ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss", color="red")
ax2.set_ylabel("F1 Macro", color="green")
ax1.tick_params(axis="y", labelcolor="red")
ax2.tick_params(axis="y", labelcolor="green")
ax2.set_ylim(0, 1.05)

ax1.legend(handles=[l1, l2, l3], loc="upper right")
ax1.set_title("Val Loss vs F1 Macro — Dual Axis", fontsize=14, fontweight="bold")
ax1.set_xticks(epochs); ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_loss_vs_f1_dual.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 04_loss_vs_f1_dual.png")

# ════════════════════════════════════════════════════════════
# BIEU DO 5: Learning Rate Schedule
# ════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(epochs, learning_rates, color="purple", linewidth=2.5, marker="o", markersize=5)
ax.fill_between(epochs, learning_rates, alpha=0.1, color="purple")

# Vùng warmup (epoch 1-4)
ax.axvspan(1, 4, alpha=0.08, color="orange", label="Warmup (ep 1-4)")
ax.axvspan(5, 16, alpha=0.05, color="blue",   label="Cosine Decay (ep 5-16)")

ax.set_title("Learning Rate Schedule: Warmup + Cosine Decay",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Epoch"); ax.set_ylabel("Learning Rate")
ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
ax.set_xticks(epochs)
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_lr_schedule.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 05_lr_schedule.png")

# ════════════════════════════════════════════════════════════
# BIEU DO 6: Confusion Matrix (Test Set — Best Epoch 11)
# ════════════════════════════════════════════════════════════
row_sum = cm.sum(axis=1, keepdims=True)
cm_norm = cm.astype(float) / row_sum

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Raw counts
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
            ax=axes[0], linewidths=0.5, linecolor="white",
            annot_kws={"size": 13, "weight": "bold"})
axes[0].set_title("Confusion Matrix (Counts)\nTest Set — Best Model (Epoch 11)",
                  fontweight="bold", fontsize=12)
axes[0].set_xlabel("Predicted Label", fontsize=11)
axes[0].set_ylabel("True Label", fontsize=11)

# Normalized
sns.heatmap(cm_norm, annot=True, fmt=".3f", cmap="YlOrRd",
            xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
            ax=axes[1], linewidths=0.5, linecolor="white",
            annot_kws={"size": 12})
axes[1].set_title("Confusion Matrix (Normalized by Row)\nRecall per Class",
                  fontweight="bold", fontsize=12)
axes[1].set_xlabel("Predicted Label", fontsize=11)
axes[1].set_ylabel("True Label", fontsize=11)

plt.suptitle("Confusion Matrix — Test Set (1338 samples, 4 Classes)",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "06_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 06_confusion_matrix.png")

# ════════════════════════════════════════════════════════════
# BIEU DO 7: Precision / Recall / F1 per Class (Bar Chart)
# ════════════════════════════════════════════════════════════
x     = np.arange(len(LABEL_NAMES))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))
b1 = ax.bar(x - width, precision_per_class, width, label="Precision",
            color="#2196F3", alpha=0.85, edgecolor="white")
b2 = ax.bar(x,          recall_per_class,   width, label="Recall",
            color="#FF9800", alpha=0.85, edgecolor="white")
b3 = ax.bar(x + width,  f1_per_class_list,  width, label="F1-Score",
            color="#4CAF50", alpha=0.85, edgecolor="white")

# Annotate
for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                f"{h:.3f}", ha="center", va="bottom",
                fontsize=8, fontweight="bold")

ax.set_title("Precision / Recall / F1 per Class — Test Set (Best Model)",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Class"); ax.set_ylabel("Score")
ax.set_xticks(x); ax.set_xticklabels(LABEL_NAMES, fontsize=12)
ax.set_ylim(0, 1.15)
ax.legend(fontsize=11); ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "07_precision_recall_f1.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 07_precision_recall_f1.png")

# ════════════════════════════════════════════════════════════
# BIEU DO 8: Training Dashboard (2x2 tổng hợp)
# ════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Panel 1: Loss
axes[0,0].plot(epochs, train_loss, "b-o", label="Train Loss", linewidth=2, markersize=4)
axes[0,0].plot(epochs, val_loss,   "r-s", label="Val Loss",   linewidth=2, markersize=4)
axes[0,0].axvline(best_epoch, color="green", linestyle="--",
                  alpha=0.8, label=f"Best ep{best_epoch}")
axes[0,0].set_title("Loss Curve", fontweight="bold")
axes[0,0].set_xlabel("Epoch"); axes[0,0].set_ylabel("Loss")
axes[0,0].legend(fontsize=9); axes[0,0].grid(True, alpha=0.3)
axes[0,0].set_xticks(epochs[::2])  # mỗi 2 epoch 1 tick

# Panel 2: Accuracy + F1
axes[0,1].plot(epochs, val_accuracy, "r-s", label="Val Accuracy", linewidth=2, markersize=4)
axes[0,1].plot(epochs, val_f1,       "g-^", label="Val F1 Macro", linewidth=2, markersize=4)
axes[0,1].axvline(best_epoch, color="purple", linestyle="--",
                  alpha=0.8, label=f"Best ep{best_epoch}")
axes[0,1].set_title("Accuracy & F1 Macro", fontweight="bold")
axes[0,1].set_xlabel("Epoch"); axes[0,1].set_ylabel("Score")
axes[0,1].set_ylim(0.3, 1.05)
axes[0,1].legend(fontsize=9); axes[0,1].grid(True, alpha=0.3)
axes[0,1].set_xticks(epochs[::2])

# Panel 3: Confusion Matrix (normalized)
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
            ax=axes[1,0], linewidths=0.5,
            annot_kws={"size": 11, "weight": "bold"})
axes[1,0].set_title("Confusion Matrix (Normalized)", fontweight="bold")
axes[1,0].set_xlabel("Predicted"); axes[1,0].set_ylabel("True")

# Panel 4: F1 per class bar
bars = axes[1,1].bar(LABEL_NAMES, f1_per_class_list,
                     color=COLORS, edgecolor="white",
                     linewidth=1.5, width=0.5)
for bar, val in zip(bars, f1_per_class_list):
    axes[1,1].text(bar.get_x() + bar.get_width()/2,
                   bar.get_height() + 0.01,
                   f"{val:.4f}", ha="center",
                   fontweight="bold", fontsize=11)
axes[1,1].set_title("F1 Score per Class (Test Set)", fontweight="bold")
axes[1,1].set_ylabel("F1 Score"); axes[1,1].set_ylim(0, 1.15)
axes[1,1].grid(True, axis="y", alpha=0.3)

plt.suptitle(
    "Training Dashboard — PhoBERT 4-Class Moderation\n"
    f"Best: Epoch {best_epoch} | F1 Macro={best_f1:.4f} | "
    f"Accuracy={val_accuracy[best_epoch-1]:.4f} | CPU Training",
    fontsize=13, fontweight="bold"
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "08_training_dashboard.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 08_training_dashboard.png")

# ════════════════════════════════════════════════════════════
# BIEU DO 9: Support Distribution (số mẫu test mỗi nhãn)
# ════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Pie chart
wedges, texts, autotexts = axes[0].pie(
    support,
    labels=LABEL_NAMES,
    colors=COLORS,
    autopct="%1.1f%%",
    startangle=90,
    pctdistance=0.75,
    wedgeprops=dict(edgecolor="white", linewidth=2)
)
for at in autotexts:
    at.set_fontsize(11)
    at.set_fontweight("bold")
axes[0].set_title("Test Set Distribution (1338 samples)", fontweight="bold")

# Bar chart support vs correct
correct = [cm[i, i] for i in range(4)]
wrong   = [support[i] - correct[i] for i in range(4)]
x = np.arange(len(LABEL_NAMES))

axes[1].bar(x, correct, color=COLORS, alpha=0.85,
            label="Correct", edgecolor="white")
axes[1].bar(x, wrong, bottom=correct, color="lightgray",
            alpha=0.7, label="Wrong", edgecolor="white")
for i, (c, w) in enumerate(zip(correct, wrong)):
    axes[1].text(i, c + w + 5, f"{support[i]}", ha="center",
                 fontweight="bold", fontsize=10)
    axes[1].text(i, c/2, f"{c}", ha="center",
                 fontweight="bold", fontsize=10, color="white")

axes[1].set_title("Correct vs Wrong Predictions per Class", fontweight="bold")
axes[1].set_xticks(x); axes[1].set_xticklabels(LABEL_NAMES)
axes[1].set_ylabel("Count"); axes[1].legend()
axes[1].grid(True, axis="y", alpha=0.3)

plt.suptitle("Test Set Analysis — Class Distribution & Prediction Quality",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "09_test_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 09_test_analysis.png")

# ════════════════════════════════════════════════════════════
print(f"\nTat ca {len(list(OUTPUT_DIR.glob('*.png')))} bieu do da luu vao:")
print(f"  {OUTPUT_DIR.resolve()}")
print("\nDanh sach:")
for f in sorted(OUTPUT_DIR.glob("*.png")):
    print(f"  {f.name}")
