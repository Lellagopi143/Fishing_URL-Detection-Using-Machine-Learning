import os

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

LEGITIMATE_FILE = "data/raw/structured_data_legitimate.csv"
PHISHING_FILE = "data/raw/structured_data_phishing.csv"

MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "phishing_url_model.pkl")


# --------------------------------------------------
# 2. Load datasets
# --------------------------------------------------

print("Loading datasets...")

legitimate_df = pd.read_csv(LEGITIMATE_FILE)
phishing_df = pd.read_csv(PHISHING_FILE)

print(f"Legitimate dataset: {legitimate_df.shape}")
print(f"Phishing dataset: {phishing_df.shape}")


# --------------------------------------------------
# 3. Combine datasets
# --------------------------------------------------

df = pd.concat(
    [legitimate_df, phishing_df],
    ignore_index=True
)

print(f"Combined dataset: {df.shape}")


# --------------------------------------------------
# 4. Remove unnecessary URL column
# --------------------------------------------------

X = df.drop(columns=["URL", "label"])
y = df["label"]


# --------------------------------------------------
# 5. Check missing values
# --------------------------------------------------

print("\nMissing values:")

missing_values = X.isnull().sum().sum()

print(f"Total missing values: {missing_values}")


# --------------------------------------------------
# 6. Train/test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------------------------
# 7. Train Random Forest
# --------------------------------------------------

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


# --------------------------------------------------
# 8. Prediction
# --------------------------------------------------

y_pred = model.predict(X_test)


# --------------------------------------------------
# 9. Evaluation
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n========== MODEL RESULTS ==========")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# --------------------------------------------------
# 10. Save model
# --------------------------------------------------

os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(model, MODEL_FILE)

print(f"\nModel saved to: {MODEL_FILE}")

print("\nTraining completed successfully!")