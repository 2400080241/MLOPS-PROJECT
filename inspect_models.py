
import joblib
from pathlib import Path

model_files = {
    "Crop Recommendation": "models/crop_recommendation_model.pkl",
    "Crop Yield": "models/crop_yield_model.pkl",
    "Irrigation": "models/irrigation_model.pkl",
    "Soil Health": "models/soil_health_model.pkl",
}

for name, filename in model_files.items():
    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    model_path = Path(filename)

    if not model_path.exists():
        print("Model file not found:", filename)
        continue

    model = joblib.load(model_path)

    print("Model type:", type(model).__name__)
    print("Features:", getattr(model, "feature_names_in_", "Unavailable"))
    print("Classes:", getattr(model, "classes_", "Not applicable"))

    if hasattr(model, "n_features_in_"):
        print("Number of features:", model.n_features_in_)

    if hasattr(model, "predict_proba"):
        print("Can calculate class probabilities: Yes")
    else:
        print("Can calculate class probabilities: No")
