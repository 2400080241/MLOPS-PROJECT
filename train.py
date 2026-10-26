import os
import joblib
import pandas as pd

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_PATH = "data/processed/agriculture_clean.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "crop_recommendation_model.pkl")

FEATURES = [
    "nitrogen",
    "phosphorus",
    "potassium",
    "temperature",
    "humidity",
    "soil_ph",
    "rainfall",
    "soil_moisture"
]

TARGET = "label"

MLFLOW_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "Smart Agriculture Crop Recommendation"


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("=" * 60)
print("SMART AGRICULTURE - MODEL TRAINING")
print("=" * 60)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)


# ============================================================
# 3. VALIDATE DATASET
# ============================================================

print("\nChecking required columns...")

required_columns = FEATURES + [TARGET]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("All required columns are present.")


# Check missing values

missing_values = df[required_columns].isnull().sum()

if missing_values.sum() > 0:
    print("\nMissing values detected:")
    print(missing_values[missing_values > 0])

    raise ValueError(
        "Dataset contains missing values. "
        "Please clean the dataset before training."
    )

print("No missing values found.")


# ============================================================
# 4. FEATURES AND TARGET
# ============================================================

X = df[FEATURES]
y = df[TARGET]

print("\nFeatures:")
print(FEATURES)

print("\nTarget:")
print(TARGET)

print("\nNumber of crop classes:", y.nunique())

print("\nCrop classes:")
print(sorted(y.unique()))


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 6. CONNECT TO MLFLOW
# ============================================================

print("\nConnecting to MLflow...")

mlflow.set_tracking_uri(MLFLOW_URI)

mlflow.set_experiment(EXPERIMENT_NAME)


# ============================================================
# 7. TRAIN RANDOM FOREST MODEL
# ============================================================

print("\nStarting model training...")

with mlflow.start_run() as run:

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    print("Model training completed.")


    # ========================================================
    # 8. PREDICTIONS
    # ========================================================

    predictions = model.predict(X_test)


    # ========================================================
    # 9. MODEL EVALUATION
    # ========================================================

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    print(f"\nAccuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions
        )
    )


    # ========================================================
    # 10. LOG PARAMETERS TO MLFLOW
    # ========================================================

    mlflow.log_param(
        "model_type",
        "RandomForestClassifier"
    )

    mlflow.log_param(
        "n_estimators",
        200
    )

    mlflow.log_param(
        "random_state",
        42
    )

    mlflow.log_param(
        "test_size",
        0.20
    )

    mlflow.log_param(
        "feature_count",
        len(FEATURES)
    )

    mlflow.log_param(
        "features",
        ",".join(FEATURES)
    )


    # ========================================================
    # 11. LOG METRICS TO MLFLOW
    # ========================================================

    mlflow.log_metric(
        "accuracy",
        accuracy
    )


    # ========================================================
    # 12. LOG MODEL TO MLFLOW
    # ========================================================

    print("\nLogging model to MLflow...")

    mlflow.sklearn.log_model(
        model,
        name="crop_recommendation_model",
        skops_trusted_types=[
            "sklearn.tree._tree.Tree"
        ]
    )

    print("Model successfully logged to MLflow.")

    print("\nMLflow Run ID:")
    print(run.info.run_id)


# ============================================================
# 13. SAVE MODEL LOCALLY
# ============================================================

print("\nSaving model locally...")

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

joblib.dump(
    model,
    MODEL_PATH
)

print("Model saved successfully:")
print(MODEL_PATH)


# ============================================================
# 14. SAMPLE FARM PREDICTION
# ============================================================

print("\n" + "=" * 60)
print("SAMPLE FARM PREDICTION")
print("=" * 60)

sample_farm = pd.DataFrame(
    [
        {
            "nitrogen": 90,
            "phosphorus": 42,
            "potassium": 43,
            "temperature": 25,
            "humidity": 80,
            "soil_ph": 6.5,
            "rainfall": 200,
            "soil_moisture": 70
        }
    ]
)

recommended_crop = model.predict(
    sample_farm
)[0]

print("\nSample farm data:")
print(sample_farm.to_string(index=False))

print("\nRecommended crop:")
print(recommended_crop)


# ============================================================
# 15. FINAL INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nDataset:")
print(DATA_PATH)

print("\nModel:")
print(MODEL_PATH)

print("\nAccuracy:")
print(f"{accuracy:.4f}")

print("\nMLflow experiment:")
print(EXPERIMENT_NAME)

print("\nMLflow server:")
print(MLFLOW_URI)