import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from glob import glob
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
from matplotlib.gridspec import GridSpec

# === CONFIGURATION ===
result_folder = 'skandhresults'
excel_files = sorted(glob(os.path.join(result_folder, '*.xlsx')))
model_names = [os.path.splitext(os.path.basename(f))[0] for f in excel_files]

# === Helper Functions ===
def plot_confusion_matrix(ax, y_true, y_pred, labels, title):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

def plot_roc(ax, y_true, y_score, labels, title):
    lb = LabelBinarizer()
    lb.fit(labels)
    y_true_binarized = lb.transform(y_true)
    y_score_binarized = lb.transform(y_score)

    fpr, tpr, roc_auc = {}, {}, {}
    for i, label in enumerate(lb.classes_):
        fpr[label], tpr[label], _ = roc_curve(y_true_binarized[:, i], y_score_binarized[:, i])
        roc_auc[label] = auc(fpr[label], tpr[label])
        ax.plot(fpr[label], tpr[label], label=f"{label} (AUC = {roc_auc[label]:.2f})")

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend()

# === DATA LOADING ===
label_list = set()
dataframes = []

for file in excel_files:
    df = pd.read_excel(file)
    df = df.rename(columns={'Image Name': 'filename',
                            'Predicted Class': 'pred_label',
                            'True Class': 'true_label'})
    label_list.update(df['true_label'].unique())
    dataframes.append(df)

labels = sorted(list(label_list))

# === CONFUSION MATRIX PLOT ===
fig_cm = plt.figure(figsize=(12, 12))
gs_cm = GridSpec(3, 3, figure=fig_cm)
axes_cm = {
    'C1': fig_cm.add_subplot(gs_cm[0, 0]),
    'C4': fig_cm.add_subplot(gs_cm[0, 2]),
    'C3': fig_cm.add_subplot(gs_cm[1, 1]),
    'C2': fig_cm.add_subplot(gs_cm[2, 0]),
    'C5': fig_cm.add_subplot(gs_cm[2, 2]),
}

for ax_key, df, model_name in zip(axes_cm.keys(), dataframes, model_names):
    ax = axes_cm[ax_key]
    plot_confusion_matrix(ax, df['true_label'], df['pred_label'], labels, title=model_name)

fig_cm.suptitle("Combined Confusion Matrices", fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.93)
fig_cm.savefig('combined_confusion_matrix_layout.png')

# === ROC CURVE PLOT ===
fig_roc = plt.figure(figsize=(12, 12))
gs_roc = GridSpec(3, 3, figure=fig_roc)
axes_roc = {
    'C1': fig_roc.add_subplot(gs_roc[0, 0]),
    'C4': fig_roc.add_subplot(gs_roc[0, 2]),
    'C3': fig_roc.add_subplot(gs_roc[1, 1]),
    'C2': fig_roc.add_subplot(gs_roc[2, 0]),
    'C5': fig_roc.add_subplot(gs_roc[2, 2]),
}

for ax_key, df, model_name in zip(axes_roc.keys(), dataframes, model_names):
    ax = axes_roc[ax_key]
    plot_roc(ax, df['true_label'], df['pred_label'], labels, title=model_name)

fig_roc.suptitle("Combined ROC Curves", fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.93)
fig_roc.savefig('combined_roc_curve_layout.png')

plt.show()
