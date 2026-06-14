"""
scripts/plot_all.py
Vẽ toàn bộ biểu đồ: Loss, Accuracy, F1, Confusion Matrix,
Per-class F1, Learning Rate, Confidence Distribution
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

# ── Đường dẫn ──────────────────────────────────────────────
HISTORY_PATH = "models/checkpoints/training_history.json"   # file lưu log mỗi epoch
OUTPUT_DIR   = Path("outputs/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_NAMES  = ["Clean", "Toxic", "Spam", "Adult"]
COLORS       = ["#4CAF50", "#F44336", "#FF9800", "#9C27B0"]

# ── Load history ────────────────────────────────────────────
with open(HISTORY_PATH, encoding="utf-8") as f:
    history = json.load(f)
# history = {
#   "train_loss": [...], "val_loss": [...],
#   "train_acc":  [...], "val_acc":  [...],
#   "val_f1_macro": [...], "val_f1_per_class": [[...], ...],
#   "learning_rate": [...],
#   "val_preds": [...], "val_labels": [...],
#   "val_confidences": [...]
# }

epochs = range(1, len(history["train_loss"]) + 1)

# ════════════════════════════════════════════════════════════
# BIỂU ĐỒ 1: Training & Validation Loss
# ════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(epochs, history["train_loss"], "b-o", label="Train Loss", linewidth=2)
ax.plot(epochs, history["val_loss"],   "r-s", label="Val Loss",   linewidth=2)
ax.set_title("📉 Training vs Validation Loss", fontsize=14, fontweight="bold")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
ax.legend(); ax.grid(True, alpha=0.3)
# Đánh dấu best epoch
best_epoch = int(np.argmin(history["val_loss"])) + 1
ax.axvline(best_epoch, color="green", linestyle="--", alpha=0.7,
           label=f"Best epoch: {best_epoch}")
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_loss_curve.png", dpi=150)
plt.close()
print("✅ Saved: 01_loss_curve.png")

# ════════════════════════════════════════════════════════════
# BIỂU ĐỒ 2: Training & Validation Accuracy
# ════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(epochs, history["train_acc"], "b-o", label="Train Accuracy", linewidth=2)
ax.plot(epochs, history["val_acc"],   "r-s", label="Val Accuracy",   linewidth=2)
ax.set_title("📈 Training vs Validation Accuracy", fontsize=14, fontweight="bold")
ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_accuracy_curve.png", dpi=150)
plt.close()
print("✅ Saved: 02_accuracy_curve.png")

# ════════════════════════════════════════════════════════════
# BIỂU ĐỒ 3: F1-Score Macro theo Epoch
# ════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(epochs, history["val_f1_macro"], "g-^", linewidth=2, label="F1 Macro")
ax.fill_between(epochs, history["val_f1_macro"], alpha=0.15, color="green")
ax.set_title("🎯 Validation F1-Score (Macro) per Epoch", fontsize=14, fontweight="bold")
ax.set_xlabel("Epoch"); ax.set_ylabel("F1 Score")
ax.grid(True, alpha=0.3); ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_f1_macro_curve.png", dpi=150)
plt.close()
print("✅ Saved: 03_f1_macro_curve.png")

# ════════════════════════════════════════════════════════════
# BIỂU ĐỒ 4: Per-Class F1 Score theo Epoch
# ════════════════════════════════════════════════════════════
f1_per_class = np.array(history["val_f1_per_class"])  # shape: [epochs, 4]
fig, ax = plt.subplots(figsize=(12, 6))
for i, (name, color) in enumerate(zip(LABEL_NAMES, COLORS)):
    ax.plot(epochs, f1_per_class[:, i], color=color,
            marker="o", linewidth=2, label=f"F1 - {name}")
ax.set_title("🏷️ Per-Class F1 Score per Epoch", fontsize=14, fontweight="bold")
ax.set_xlabel("Epoch"); ax.set_ylabel("F1 Score")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_f1_per_class_curve.png", dpi=150)
plt.close()
print("✅ Saved: 04_f1_per_class_curve.png")

# ════════════════════════════════════════════════════════════
# BIỂU ĐỒ 5: Confusion Matrix (Best Epoch / Final)
# ════════════════════════════════════════════════════════════
from sklearn.metrics import confusion_matrix

y_true = history["val_labels"]   # list of int
y_pred = history["val_preds"]    # list of int

cm = confusion_matrix(y_true, y_pred, labels=[0,1,2,3])
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)  # normalize

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Raw counts
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=axes[0])
axes[0].set_title("Confusion Matrix (Counts)", fontweight="bold")
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")

# Normalized
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="YlOrRd",
            xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=axes[1])
axes[1].set_title("Confusion Matrix (Normalized)", fontweight="bold")
axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")

plt.suptitle("🔢 Confusion Matrix — Final Evaluation", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_confusion_matrix.png", dpi=150)
plt.close()
print("✅ Saved: 05_confusion_matrix.png")

# ════════════════════════════════════════════════════════════
# BIỂU ĐỒ 6: Learning Rate Schedule
# ════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(history["learning_rate"], color="purple", linewidth=2)
ax.set_title("📐 Learning Rate Schedule (Warmup + Cosine Decay)",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Step"); ax.set_ylabel("Learning Rate")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "06_lr_schedule.png", dpi=150)
plt.close()
print("✅ Saved: 06_lr_schedule.png")

# ════════════════════════════════════════════════════════════
# BIỂU ĐỒ 7: Confidence Score Distribution per Class
# ════════════════════════════════════════════════════════════
confidences = np.array(history["val_confidences"])  # shape: [N, 4]
labels_true  = np.array(history["val_labels"])

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
for i, (name, color) in enumerate(zip(LABEL_NAMES, COLORS)):
    mask = labels_true == i
    conf_for_class = confidences[mask, i]
    axes[i].hist(conf_for_class, bins=30, color=color, alpha=0.75, edgecolor="white")
    axes[i].axvline(conf_for_class.mean(), color="black", linestyle="--",
                    label=f"Mean: {conf_for_class.mean():.2f}")
    axes[i].set_title(f"Confidence Distribution — {name}", fontweight="bold")
    axes[i].set_xlabel("Confidence Score"); axes[i].set_ylabel("Count")
    axes[i].legend(); axes[i].grid(True, alpha=0.3)

plt.suptitle("📊 Model Confidence Distribution per Class",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "07_confidence_distribution.png", dpi=150)
plt.close()
print("✅ Saved: 07_confidence_distribution.png")

# ════════════════════════════════════════════════════════════
# BIỂU ĐỒ 8: Bar Chart — Final F1 per Class
# ════════════════════════════════════════════════════════════
final_f1 = f1_per_class[-1]  # epoch cuối
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(LABEL_NAMES, final_f1, color=COLORS, edgecolor="white", linewidth=1.5)
for bar, val in zip(bars, final_f1):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", fontweight="bold")
ax.set_title("🏆 Final F1 Score per Class", fontsize=14, fontweight="bold")
ax.set_ylabel("F1 Score"); ax.set_ylim(0, 1.1)
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "08_final_f1_bar.png", dpi=150)
plt.close()
print("✅ Saved: 08_final_f1_bar.png")

# ════════════════════════════════════════════════════════════
# BIỂU ĐỒ 9: Loss + F1 kết hợp (dual axis)
# ════════════════════════════════════════════════════════════
fig, ax1 = plt.subplots(figsize=(12, 5))
ax2 = ax1.twinx()
ax1.plot(epochs, history["val_loss"],     "r-o", label="Val Loss",    linewidth=2)
ax2.plot(epochs, history["val_f1_macro"], "g-s", label="Val F1 Macro",linewidth=2)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss",     color="red")
ax2.set_ylabel("F1 Macro", color="green")
ax1.tick_params(axis="y", labelcolor="red")
ax2.tick_params(axis="y", labelcolor="green")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
ax1.set_title("📉📈 Val Loss vs F1 Macro (Dual Axis)", fontsize=14, fontweight="bold")
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "09_loss_vs_f1_dual.png", dpi=150)
plt.close()
print("✅ Saved: 09_loss_vs_f1_dual.png")

print(f"\n🎉 Tất cả biểu đồ đã lưu vào: {OUTPUT_DIR.resolve()}")
