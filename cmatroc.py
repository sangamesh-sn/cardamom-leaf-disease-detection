import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from glob import glob

# === CONFIGURATION ===
image_root = 'dataset/test'
result_file = 'test_results_mnv2.xlsx'
cm_plot_file = 'confusion_matrix.png'
roc_plot_file = 'roc_curve.png'
grid_plot_file = 'annotated_grid.png'
supported_formats = ('*.jpg', '*.jpeg', '*.png')

# === STEP 1: LOAD PREDICTIONS ===
df = pd.read_excel(result_file)

# Rename columns for consistency
df = df.rename(columns={
    'Image Name': 'filename',
    'Predicted Class': 'pred_label',
    'True Class': 'true_label'
})

true_labels = df['true_label']
pred_labels = df['pred_label']
classes = sorted(true_labels.unique())
n_classes = len(classes)

# Label encoding
le = LabelEncoder()
true_encoded = le.fit_transform(true_labels)
pred_encoded = le.transform(pred_labels)

# === STEP 2: CONFUSION MATRIX ===
cm = confusion_matrix(true_encoded, pred_encoded)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap='YlGnBu', xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(cm_plot_file, dpi=300)
plt.close()
print(f"[✅] Confusion matrix saved to {cm_plot_file}")

# === STEP 3: ROC CURVE ===
lb = LabelBinarizer()
y_true_bin = lb.fit_transform(true_labels)
y_pred_bin = lb.transform(pred_labels)

plt.figure(figsize=(8, 6))
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{classes[i]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(roc_plot_file, dpi=300)
plt.close()
print(f"[✅] ROC curve saved to {roc_plot_file}")

# === STEP 4: ANNOTATED IMAGE GRID ===
image_paths = []
for ext in supported_formats:
    image_paths.extend(glob(os.path.join(image_root, '**', ext), recursive=True))

# Match filename from dataframe with available images
annotated_images = []
for path in image_paths:
    filename = os.path.basename(path)
    match = df[df['filename'] == filename]
    if not match.empty:
        true_label = match.iloc[0]['true_label']
        pred_label = match.iloc[0]['pred_label']
        annotated_images.append((path, true_label, pred_label))

plt.figure(figsize=(4 * n_classes, 4 * n_classes))
count = 1
used_pairs = set()

for true_class in classes:
    for pred_class in classes:
        found = False
        for (path, t, p) in annotated_images:
            if (t == true_class) and (p == pred_class) and ((t, p) not in used_pairs):
                img = cv2.imread(path)
                if img is None:
                    print(f"[Warning] Could not load image: {path}")
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                used_pairs.add((t, p))

                plt.subplot(n_classes, n_classes, count)
                plt.imshow(img)
                plt.axis('off')

                color = 'green' if t == p else 'red'
                plt.title(f"True: {t}\nPred: {p}", fontsize=10, color=color)
                count += 1
                found = True
                break
        if not found:
            plt.subplot(n_classes, n_classes, count)
            plt.axis('off')
            plt.title(f"True: {true_class}\nPred: {pred_class}", fontsize=10, color='gray')
            count += 1

plt.tight_layout()
plt.savefig(grid_plot_file, dpi=300)
plt.close()
print(f"[✅] Annotated grid saved to {grid_plot_file}")
