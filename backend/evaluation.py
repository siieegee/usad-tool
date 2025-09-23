import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import pandas as pd

# =========================================
# 1. Load your test dataset
# =========================================
df = pd.read_csv("clustered_reviews_test.csv")

# Map labels explicitly:
# OR = 0 (Normal), CG = 1 (Anomalous)
df['true_anomaly'] = df['label'].map({'OR': 0, 'CG': 1})

# Model predictions:
# review_type column contains either 'Normal' or 'Anomalous'
df['predicted_anomaly'] = df['review_type'].map({'Normal': 0, 'Anomalous': 1})

# Extract arrays for sklearn metrics
y_test_true = df['true_anomaly'].values
y_pred = df['predicted_anomaly'].values

print("Data loaded and mapped successfully!\n")
print(df[['label', 'review_type', 'true_anomaly', 'predicted_anomaly']].head())

# =========================================
# 2. Compute confusion matrix
# =========================================
cm = confusion_matrix(y_test_true, y_pred)
print("\nConfusion Matrix (Raw Numbers):")
print(cm)

# Break down the matrix into individual components
tn, fp, fn, tp = cm.ravel()

print("\nBreakdown of Confusion Matrix:")
print(f"True Negatives (TN): {tn} -> Correctly identified Normal (OR)")
print(f"False Positives (FP): {fp} -> Normal (OR) incorrectly flagged as Anomalous (CG)")
print(f"False Negatives (FN): {fn} -> Anomalous (CG) missed and labeled Normal (OR)")
print(f"True Positives (TP): {tp} -> Correctly identified Anomalous (CG)")

# =========================================
# 3. Compute and print performance metrics
# =========================================
accuracy = accuracy_score(y_test_true, y_pred)
precision = precision_score(y_test_true, y_pred, zero_division=0)
recall = recall_score(y_test_true, y_pred, zero_division=0)
f1 = f1_score(y_test_true, y_pred, zero_division=0)

print("\nAdditional Performance Metrics:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")

# =========================================
# 4. Print full classification report
# =========================================
print("\nClassification Report:")
print(classification_report(
    y_test_true,
    y_pred,
    target_names=["Normal (OR)", "Anomalous (CG)"]
))

# =========================================
# 5. Display Confusion Matrix Visualization
# =========================================
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal (OR)", "Anomalous (CG)"])
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix")
plt.show()
