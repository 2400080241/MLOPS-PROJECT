
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

MODEL_DIR = "models"

FEATURES = [
    "temperature",
    "humidity",
    "soil_moisture",
    "rainfall",
    "soil_ph",
    "nitrogen",
    "phosphorus",
    "potassium",
]


def load_model(filename):
    path = os.path.join(MODEL_DIR, filename)

    if not os.path.exists(path):
        print(f"\nModel file not found: {path}")
        return None

    return joblib.load(path)


def prepare_features(model, X):
    """Arrange input features in the order expected by the model."""
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)

        missing = [column for column in expected if column not in X.columns]
        if missing:
            raise ValueError(
                f"Missing model input features: {missing}"
            )

        return X[expected]

    if hasattr(model, "n_features_in_") and model.n_features_in_ != X.shape[1]:
        raise ValueError(
            f"Model expects {model.n_features_in_} features, "
            f"but received {X.shape[1]}."
        )

    return X


def evaluate_classifier(name, model, X, y):
    X_model = prepare_features(model, X)
    predictions = model.predict(X_model)

    print(f"\n{'=' * 55}")
    print(name)
    print(f"{'=' * 55}")
    print(f"Samples evaluated: {len(y)}")
    print(f"Accuracy: {accuracy_score(y, predictions):.4f}")
    print(
        "Balanced accuracy: "
        f"{balanced_accuracy_score(y, predictions):.4f}"
    )
    print("\nClassification report:")
    print(classification_report(y, predictions, zero_division=0))


def main():
    print("SMART AGRICULTURE MODEL EVALUATION")

    # 1. Crop recommendation
    crop_path = "data/processed/agriculture_clean.csv"

    if os.path.exists(crop_path):
        crop_df = pd.read_csv(crop_path)

        if "label" in crop_df.columns and all(
            column in crop_df.columns for column in FEATURES
        ):
            crop_df = crop_df.dropna(subset=FEATURES + ["label"])
            model = load_model("crop_recommendation_model.pkl")

            if model is not None:
                evaluate_classifier(
                    "CROP RECOMMENDATION",
                    model,
                    crop_df[FEATURES],
                    crop_df["label"],
                )
        else:
            print("\nCrop dataset is missing required columns.")
            print("Available columns:", crop_df.columns.tolist())
    else:
        print(f"\nCrop dataset not found: {crop_path}")

    # 2. Yield, soil health, and irrigation
    data_path = "data/agriculture.csv"

    if not os.path.exists(data_path):
        print(f"\nDataset not found: {data_path}")
        return

    df = pd.read_csv(data_path)

    targets = [
        "crop_yield",
        "soil_health",
        "irrigation_required",
    ]
    required = FEATURES + targets

    missing = [column for column in required if column not in df.columns]

    if missing:
        print("\nMissing dataset columns:", missing)
        return

    df = df.dropna(subset=required)
    X = df[FEATURES]

    # 3. Crop yield regression
    model = load_model("crop_yield_model.pkl")

    if model is not None:
        X_yield = prepare_features(model, X)
        actual = df["crop_yield"]
        predicted = model.predict(X_yield)

        print(f"\n{'=' * 55}")
        print("CROP YIELD")
        print(f"{'=' * 55}")
        print(f"Samples evaluated: {len(actual)}")
        print(f"MAE: {mean_absolute_error(actual, predicted):.4f}")
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        print(f"RMSE: {rmse:.4f}")
        print(f"R-squared: {r2_score(actual, predicted):.4f}")
        print("Yield units are unchanged from your dataset.")

    # 4. Soil health classification
    model = load_model("soil_health_model.pkl")

    if model is not None:
        evaluate_classifier(
            "SOIL HEALTH",
            model,
            X,
            df["soil_health"],
        )

    # 5. Irrigation classification
    model = load_model("irrigation_model.pkl")

    if model is not None:
        evaluate_classifier(
            "IRRIGATION REQUIREMENT",
            model,
            X,
            df["irrigation_required"],
        )


if __name__ == "__main__":
    main()
