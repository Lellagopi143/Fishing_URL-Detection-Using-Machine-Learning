import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


# ============================================================
# 1. CONFIGURATION
# ============================================================

LEGITIMATE_FILE = "data/raw/structured_data_legitimate.csv"
PHISHING_FILE = "data/raw/structured_data_phishing.csv"

MODEL_DIR = "models"
MODEL_FILE = os.path.join(
    MODEL_DIR,
    "phishing_url_model.pkl"
)

# Model parameters
N_ESTIMATORS = 200
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Minimum accuracy required for the model to pass
MIN_ACCURACY = 0.90


# ============================================================
# 2. MLflow CONFIGURATION
# ============================================================

mlflow.set_experiment(
    "Phishing_URL_Detection"
)


# ============================================================
# 3. LOAD DATASET
# ============================================================

def load_data():
    print("\n========================================")
    print("LOADING DATASETS")
    print("========================================")

    legitimate_df = pd.read_csv(
        LEGITIMATE_FILE
    )

    phishing_df = pd.read_csv(
        PHISHING_FILE
    )

    print(
        f"Legitimate dataset shape: "
        f"{legitimate_df.shape}"
    )

    print(
        f"Phishing dataset shape: "
        f"{phishing_df.shape}"
    )

    return legitimate_df, phishing_df


# ============================================================
# 4. COMBINE DATASETS
# ============================================================

def prepare_data(
    legitimate_df,
    phishing_df
):

    print("\n========================================")
    print("PREPARING DATA")
    print("========================================")

    df = pd.concat(
        [
            legitimate_df,
            phishing_df
        ],
        ignore_index=True
    )

    print(
        f"Combined dataset shape: {df.shape}"
    )

    # Check required columns
    required_columns = [
        "URL",
        "label"
    ]

    for column in required_columns:

        if column not in df.columns:
            raise ValueError(
                f"Required column '{column}' "
                f"not found in dataset."
            )

    # Separate features and target
    #
    # URL is removed because the model is using
    # the engineered URL features already present
    # in the dataset.

    X = df.drop(
        columns=["URL", "label"]
    )

    y = df["label"]

    print(
        f"Number of features: {X.shape[1]}"
    )

    print("\nClass distribution:")

    print(y.value_counts())

    # Check missing values
    missing_values = X.isnull().sum().sum()

    print(
        f"\nTotal missing feature values: "
        f"{missing_values}"
    )

    if missing_values > 0:

        print(
            "Missing values detected. "
            "Filling with 0."
        )

        X = X.fillna(0)

    return X, y


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

def split_data(X, y):

    print("\n========================================")
    print("SPLITTING DATA")
    print("========================================")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(
        f"Training samples: {len(X_train)}"
    )

    print(
        f"Testing samples: {len(X_test)}"
    )

    return X_train, X_test, y_train, y_test


# ============================================================
# 6. TRAIN MODEL
# ============================================================

def train_model(
    X_train,
    y_train
):

    print("\n========================================")
    print("TRAINING RANDOM FOREST")
    print("========================================")

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    print(
        "Model training completed."
    )

    return model


# ============================================================
# 7. EVALUATE MODEL
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test
):

    print("\n========================================")
    print("MODEL EVALUATION")
    print("========================================")

    y_pred = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    print(
        f"\nAccuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    print("Confusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    return metrics, y_pred


# ============================================================
# 8. MODEL QUALITY GATE
# ============================================================

def quality_gate(accuracy):

    print("\n========================================")
    print("MODEL QUALITY GATE")
    print("========================================")

    print(
        f"Required accuracy: "
        f"{MIN_ACCURACY:.2%}"
    )

    print(
        f"Actual accuracy: "
        f"{accuracy:.2%}"
    )

    if accuracy < MIN_ACCURACY:

        print(
            "\n❌ MODEL QUALITY GATE FAILED"
        )

        print(
            "Model will NOT be saved as "
            "a production model."
        )

        raise ValueError(
            f"Model accuracy "
            f"{accuracy:.4f} is below "
            f"the required threshold "
            f"{MIN_ACCURACY:.4f}."
        )

    print(
        "\n✅ MODEL QUALITY GATE PASSED"
    )


# ============================================================
# 9. SAVE MODEL
# ============================================================

def save_model(model):

    print("\n========================================")
    print("SAVING MODEL")
    print("========================================")

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_FILE
    )

    print(
        f"Model saved successfully:"
    )

    print(
        MODEL_FILE
    )


# ============================================================
# 10. MAIN TRAINING PIPELINE
# ============================================================

def main():

    print("\n")
    print("========================================")
    print("PHISHING URL DETECTION")
    print("MLOps TRAINING PIPELINE")
    print("========================================")

    # --------------------------------------------------------
    # Start MLflow run
    # --------------------------------------------------------

    with mlflow.start_run():

        # ----------------------------------------------------
        # Load datasets
        # ----------------------------------------------------

        legitimate_df, phishing_df = load_data()

        # ----------------------------------------------------
        # Prepare data
        # ----------------------------------------------------

        X, y = prepare_data(
            legitimate_df,
            phishing_df
        )

        # ----------------------------------------------------
        # Train/test split
        # ----------------------------------------------------

        X_train, X_test, y_train, y_test = split_data(
            X,
            y
        )

        # ----------------------------------------------------
        # Log MLflow parameters
        # ----------------------------------------------------

        mlflow.log_param(
            "model",
            "RandomForestClassifier"
        )

        mlflow.log_param(
            "n_estimators",
            N_ESTIMATORS
        )

        mlflow.log_param(
            "random_state",
            RANDOM_STATE
        )

        mlflow.log_param(
            "test_size",
            TEST_SIZE
        )

        mlflow.log_param(
            "number_of_features",
            X.shape[1]
        )

        mlflow.log_param(
            "training_samples",
            len(X_train)
        )

        mlflow.log_param(
            "testing_samples",
            len(X_test)
        )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        model = train_model(
            X_train,
            y_train
        )

        # ----------------------------------------------------
        # Evaluate
        # ----------------------------------------------------

        metrics, y_pred = evaluate_model(
            model,
            X_test,
            y_test
        )

        # ----------------------------------------------------
        # Log MLflow metrics
        # ----------------------------------------------------

        mlflow.log_metric(
            "accuracy",
            metrics["accuracy"]
        )

        mlflow.log_metric(
            "precision",
            metrics["precision"]
        )

        mlflow.log_metric(
            "recall",
            metrics["recall"]
        )

        mlflow.log_metric(
            "f1_score",
            metrics["f1_score"]
        )

        # ----------------------------------------------------
        # Quality gate
        # ----------------------------------------------------

        quality_gate(
            metrics["accuracy"]
        )

        # ----------------------------------------------------
        # Save model locally
        # ----------------------------------------------------

        save_model(
            model
        )

        # ----------------------------------------------------
        # Log model to MLflow
        # ----------------------------------------------------

        mlflow.sklearn.log_model(
            sk_model=model,
            name="phishing_url_model",
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ]
        )

        print(
            "\nModel logged successfully "
            "to MLflow."
        )

        # ----------------------------------------------------
        # Final information
        # ----------------------------------------------------

        print("\n========================================")
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("========================================")

        print(
            f"Accuracy : "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"Precision: "
            f"{metrics['precision']:.4f}"
        )

        print(
            f"Recall   : "
            f"{metrics['recall']:.4f}"
        )

        print(
            f"F1 Score : "
            f"{metrics['f1_score']:.4f}"
        )

        print(
            f"\nModel: {MODEL_FILE}"
        )


# ============================================================
# 11. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()