import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, precision_score,
    recall_score, f1_score, classification_report
)

# === FILE PATHS ===
RESULT_FILE = "test_results_mnv2.xlsx"
HISTORY_FILE = "training_history_mnv2_3classes.xlsx"
OUTPUT_FILE = "combined_confusion_matrix_layout.png"


# === LOAD DATA ===
if not os.path.exists(RESULT_FILE):
    raise FileNotFoundError(f"Result file not found: {RESULT_FILE}")

if not os.path.exists(HISTORY_FILE):
    raise FileNotFoundError(f"History file not found: {HISTORY_FILE}")

results_df = pd.read_excel(RESULT_FILE)
history_df = pd.read_excel(HISTORY_FILE)

true_col = "True Class"
pred_col = "Predicted Class"

if true_col not in results_df.columns or pred_col not in results_df.columns:
    raise ValueError(f"Columns not found in Excel. Found: {results_df.columns}")

y_true = results_df[true_col].values
y_pred = results_df[pred_col].values

class_names = sorted(results_df[true_col].unique())

# Convert class names to indices
label_to_idx = {label: idx for idx, label in enumerate(class_names)}
y_true_idx = np.array([label_to_idx[x] for x in y_true])
y_pred_idx = np.array([label_to_idx[x] for x in y_pred])

# === CONFUSION MATRIX ===
cm = confusion_matrix(y_true_idx, y_pred_idx)

# === PRECISION / RECALL / F1 ===
precision = precision_score(y_true_idx, y_pred_idx, average=None)
recall = recall_score(y_true_idx, y_pred_idx, average=None)
f1 = f1_score(y_true_idx, y_pred_idx, average=None)

# === PLOTTING ===
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
ax_cm, ax_metrics, ax_loss, ax_acc = axes.flatten()

# 1) CONFUSION MATRIX
sns.heatmap(
    cm,
    annot=True,
    cmap="Blues",
    fmt="d",
    xticklabels=class_names,
    yticklabels=class_names,
    ax=ax_cm
)
ax_cm.set_title("Confusion Matrix")
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("True")

# 2) PRECISION / RECALL / F1 BAR CHART
x = np.arange(len(class_names))
width = 0.25

ax_metrics.bar(x - width, precision * 100, width, label="Precision")
ax_metrics.bar(x, recall * 100, width, label="Recall")
ax_metrics.bar(x + width, f1 * 100, width, label="F1 Score")

ax_metrics.set_xticks(x)
ax_metrics.set_xticklabels(class_names, rotation=15)
ax_metrics.set_ylabel("Percentage (%)")
ax_metrics.set_title("Precision / Recall / F1-Score")
ax_metrics.legend()

# 3) LOSS CURVES
if "loss" in history_df.columns and "val_loss" in history_df.columns:
    ax_loss.plot(history_df["loss"], label="Train Loss")
    ax_loss.plot(history_df["val_loss"], label="Val Loss")
    ax_loss.set_title("Loss vs Epochs")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()
else:
    ax_loss.text(0.5, 0.5, "Loss data missing", ha='center')
    
# === PRINT CLASSIFICATION REPORT (precision, recall, f1, support) ===
print("\n================ Classification Report ================\n")

report = classification_report(
    y_true,
    y_pred,
    labels=class_names,        # ensure order
    target_names=class_names,  # row names
    digits=4
)

print(report)
print("========================================================\n")


# 4) ACCURACY CURVES
if "accuracy" in history_df.columns and "val_accuracy" in history_df.columns:
    ax_acc.plot(history_df["accuracy"], label="Train Accuracy")
    ax_acc.plot(history_df["val_accuracy"], label="Val Accuracy")
    ax_acc.set_title("Accuracy vs Epochs")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.legend()
else:
    ax_acc.text(0.5, 0.5, "Accuracy data missing", ha='center')

plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=300)
plt.close()

print(f"\n[✅] Combined report saved → {OUTPUT_FILE}\n")