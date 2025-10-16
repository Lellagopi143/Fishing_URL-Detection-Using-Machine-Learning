import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ===========================
# Step 1: Load datasets
# ===========================
phishing_df = pd.read_csv("structured_data_phishing.csv")
legitimate_df = pd.read_csv("structured_data_legitimate.csv")

# ===========================
# Step 2: Combine datasets
# ===========================
phishing_df["label"] = 1
legitimate_df["label"] = 0

data = pd.concat([phishing_df, legitimate_df], axis=0).reset_index(drop=True)
print("Combined dataset shape:", data.shape)
print(data["label"].value_counts())

# ===========================
# Step 3: Drop URL column and separate features & labels
# ===========================
X = data.drop(["URL", "label"], axis=1)  # numeric features only
y = data["label"]

print("Features shape:", X.shape)

# ===========================
# Step 4: Train-test split
# ===========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===========================
# Step 5: Scale features
# ===========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===========================
# Step 6: Train SVM
# ===========================
svm_model = SVC(kernel="rbf", C=1, gamma="scale", probability=True, random_state=42)
svm_model.fit(X_train, y_train)

# ===========================
# Step 7: Predict & Evaluate
# ===========================
y_pred = svm_model.predict(X_test)

print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, zero_division=0, digits=4))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Legitimate (0)", "Phishing (1)"],
            yticklabels=["Legitimate (0)", "Phishing (1)"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - SVM")
plt.show()

# ===========================
# Step 8: Save Model & Scaler
# ===========================
joblib.dump(svm_model, "svm_model.pkl")
joblib.dump(scaler, "scaler_fishing.pkl")