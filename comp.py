import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ======= CONFIG: filenames from your training scripts =======
XCEPTION_HISTORY_FILE = "training_history_Xception_3classes.xlsx"
MNV2_HISTORY_FILE     = "training_history_mnv2_3classes.xlsx"

OUTPUT_FIG = "model_accuracy_comparison.png"
# ============================================================

# ---- Load test accuracy for each model from Excel ----
x_hist = pd.read_excel(XCEPTION_HISTORY_FILE)
m_hist = pd.read_excel(MNV2_HISTORY_FILE)

# Your training scripts stored a single value in 'test_accuracy'
x_test_acc = float(x_hist["test_accuracy"].iloc[0]) * 100.0   # to %
m_test_acc = float(m_hist["test_accuracy"].iloc[0]) * 100.0   # to %

models = ["Xception", "MobileNetV2"]
means = np.array([x_test_acc, m_test_acc])

# If you had multiple runs, you would compute std dev here.
# For now we show small dummy error bars (you can change these if needed):
stds = np.array([2.0, 2.0])   # +/- 2% error bars just for visual comparison

print("\nModel Test Accuracies (%)")
print("--------------------------")
print(f"Xception     : {x_test_acc:.2f}%")
print(f"MobileNetV2  : {m_test_acc:.2f}%\n")

# ---- Plot bar chart ----
plt.figure(figsize=(6, 6))

bars = plt.bar(models, means, yerr=stds, capsize=10)

plt.ylabel("Mean Test Accuracy (%)")
plt.xlabel("DL Models")
plt.title("Simple Bar Mean of Test Accuracy (%) by DL Models")

# Optional: add value labels on top of bars
for bar, val in zip(bars, means):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 1,
             f"{val:.1f}%", ha="center", va="bottom", fontsize=10)

plt.ylim(0, 120)  # like in your example figure (0–150, adjust if you want)

plt.tight_layout()
plt.savefig(OUTPUT_FIG, dpi=300)
plt.close()

print(f"[✅] Comparison figure saved to '{OUTPUT_FIG}'")