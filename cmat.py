import os
from glob import glob

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    classification_report
)
from sklearn.preprocessing import LabelBinarizer, LabelEncoder

# ============== CONFIGURATION ==============

# Root folder that contains your test images
image_root = "dataset/test"

# Excel file created by your Xception script
result_file = "test_results_mnv2.xlsx"

# Output image filenames
cm_plot_file = "mobilenetv2_confusion_matrix.png"
roc_plot_file = "mobilenetv2_roc_curve.png"
grid_plot_file = "mobilenetv2_annotated_grid.png"

# Supported image extensions (for annotated grid search)
supported_formats = (".jpg", ".jpeg", "*.png")

# ============== STEP 1: LOAD PREDICTIONS ==============

if not os.path.exists(result_file):
    raise FileNotFoundError(f"Prediction file not found: {result_file}")

df = pd.read_excel(result_file)

# Expect these columns from your training script
# "Image Name", "Predicted Class", "True Class", "Accuracy"
expected_cols = ["Image Name", "Predicted Class", "True Class"]
for col in expected_cols:
    if col not in df.columns:
        raise ValueError(
            f"Column '{col}' not found in {result_file}. "
            f"Available columns: {list(df.columns)}"
        )

# Rename for convenience
df = df.rename(
    columns={
        "Image Name": "filename",
        "Predicted Class": "pred_label",
        "True Class": "true_label",
    }
)

true_labels = df["true_label"].values
pred_labels = df["pred_label"].values

class_names = sorted(df["true_label"].unique().tolist())
n_classes = len(class_names)

print("\nClasses detected:", class_names)

# Encode string labels -> integer indices
le = LabelEncoder()
y_true_idx = le.fit_transform(true_labels)
# Use same mapping for predicted labels
y_pred_idx = le.transform(pred_labels)

# ============== STEP 2: PRINT PRECISION / RECALL / F1 / SUPPORT ==============

print("\n================ Classification Report ================\n")

report = classification_report(
    y_true_idx,
    y_pred_idx,
    target_names=class_names,
    digits=4,
)
print(report)
print("========================================================\n")

# ============== STEP 3: CONFUSION MATRIX ==============

cm = confusion_matrix(y_true_idx, y_pred_idx)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("mobilenetv2 Confusion Matrix")
plt.tight_layout()
plt.savefig(cm_plot_file, dpi=300)
plt.close()

print(f"[✅] Confusion matrix saved to {cm_plot_file}")

# ============== STEP 4: ROC CURVES (One-vs-Rest) ==============

# Binarize labels for multi-class ROC
lb = LabelBinarizer()
y_true_bin = lb.fit_transform(y_true_idx)
y_pred_bin = lb.transform(y_pred_idx)  # using predicted labels as scores (0/1)

plt.figure(figsize=(8, 6))

for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {roc_auc:.2f})")

# Diagonal line
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("mobilenetv2 ROC Curves (One-vs-Rest)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(roc_plot_file, dpi=300)
plt.close()

print(f"[✅] ROC curve saved to {roc_plot_file}")

# ============== STEP 5: ANNOTATED IMAGE GRID ==============

def find_image_path(root, filename):
    """
    Search for an image file with this filename under 'root' recursively.
    """
    for pattern in supported_formats:
        matches = glob(os.path.join(root, "**", filename), recursive=True)
        if matches:
            return matches[0]
    return None

# Take up to 16 images for the grid
grid_df = df.copy().reset_index(drop=True)
max_images = min(16, len(grid_df))
grid_df = grid_df.iloc[:max_images]

n_rows = 4
n_cols = 4

plt.figure(figsize=(12, 12))

for idx, row in grid_df.iterrows():
    filename = row["filename"]
    true_label = row["true_label"]
    pred_label = row["pred_label"]

    img_path = find_image_path(image_root, filename)

    if img_path is None:
        # Skip if image not found
        continue

    img = cv2.imread(img_path)
    if img is None:
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.subplot(n_rows, n_cols, idx + 1)
    plt.imshow(img)
    plt.axis("off")

    color = "green" if true_label == pred_label else "red"
    plt.title(f"T: {true_label}\nP: {pred_label}", color=color, fontsize=10)

plt.suptitle("mobilenetv2 – Annotated Test Images (T=True, P=Predicted)", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(grid_plot_file, dpi=300)
plt.close()

print(f"[✅] Annotated grid saved to {grid_plot_file}\n")