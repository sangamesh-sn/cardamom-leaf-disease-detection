import pandas as pd
from sklearn.metrics import classification_report

# === FILE WITH PREDICTIONS ===
RESULT_FILE = "test_results_mnv2.xlsx"   # adjust if your file name is different

# === LOAD DATA ===
df = pd.read_excel(RESULT_FILE)

TRUE_COL = "True Class"
PRED_COL = "Predicted Class"

if TRUE_COL not in df.columns or PRED_COL not in df.columns:
    raise ValueError(
        f"Expected columns '{TRUE_COL}' and '{PRED_COL}' in {RESULT_FILE}, "
        f"but got: {list(df.columns)}"
    )

y_true = df[TRUE_COL].values
y_pred = df[PRED_COL].values

# Get class names (sorted for consistent order)
class_names = sorted(df[TRUE_COL].unique().tolist())

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