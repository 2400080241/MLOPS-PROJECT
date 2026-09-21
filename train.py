import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==================================================
# LOAD DATA
# ==================================================

df = pd.read_csv("data/agriculture.csv")

print("Dataset shape:", df.shape)


# ==================================================
# VALIDATE DATA
# ==================================================

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())


# ==================================================
# INPUT FEATURES
# ==================================================

features = [
    "temperature",
    "humidity",
    "soil_moisture",
    "rainfall",
    "soil_ph",
    "nitrogen",
    "phosphorus",
    "potassium"
]

X = df[features]


# ==================================================
# MODEL 1: IRRIGATION
# ==================================================

print("\n")
print("========================================")
print("IRRIGATION PREDICTION")
print("========================================")

y_irrigation = df["irrigation_required"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_irrigation,
    test_size=0.20,
    random_state=42,
    stratify=y_irrigation
)

irrigation_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

irrigation_model.fit(X_train, y_train)

irrigation_prediction = irrigation_model.predict(X_test)

irrigation_accuracy = accuracy_score(
    y_test,
    irrigation_prediction
)

print("Accuracy:", irrigation_accuracy)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        irrigation_prediction
    )
)


# ==================================================
# MODEL 2: SOIL HEALTH
# ==================================================

print("\n")
print("========================================")
print("SOIL HEALTH PREDICTION")
print("========================================")

y_soil = df["soil_health"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_soil,
    test_size=0.20,
    random_state=42,
    stratify=y_soil
)

soil_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

soil_model.fit(X_train, y_train)

soil_prediction = soil_model.predict(X_test)

soil_accuracy = accuracy_score(
    y_test,
    soil_prediction
)

print("Accuracy:", soil_accuracy)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        soil_prediction
    )
)


# ==================================================
# MODEL 3: CROP YIELD
# ==================================================

print("\n")
print("========================================")
print("CROP YIELD PREDICTION")
print("========================================")

y_yield = df["crop_yield"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_yield,
    test_size=0.20,
    random_state=42
)

yield_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

yield_model.fit(X_train, y_train)

yield_prediction = yield_model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    yield_prediction
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        yield_prediction
    )
)

r2 = r2_score(
    y_test,
    yield_prediction
)

print("MAE :", mae)
print("RMSE:", rmse)
print("R²  :", r2)


# ==================================================
# ACTUAL VS PREDICTED
# ==================================================

print("\n")
print("========================================")
print("CROP YIELD - ACTUAL VS PREDICTED")
print("========================================")

for actual, predicted in zip(
    y_test,
    yield_prediction
):

    print(
        "Actual:",
        round(actual, 2),
        " | Predicted:",
        round(predicted, 2)
    )


# ==================================================
# NEW FARM PREDICTION
# ==================================================

print("\n")
print("========================================")
print("NEW FARM PREDICTION")
print("========================================")


new_farm = pd.DataFrame(
    {
        "temperature": [35],
        "humidity": [55],
        "soil_moisture": [38],
        "rainfall": [0],
        "soil_ph": [5.9],
        "nitrogen": [52],
        "phosphorus": [29],
        "potassium": [24]
    }
)


# Irrigation prediction

irrigation_result = irrigation_model.predict(
    new_farm
)[0]


# Soil health prediction

soil_result = soil_model.predict(
    new_farm
)[0]


# Crop yield prediction

yield_result = yield_model.predict(
    new_farm
)[0]


# Display irrigation

if irrigation_result == 1:
    print("Irrigation: REQUIRED")
else:
    print("Irrigation: NOT REQUIRED")


# Display soil health

if soil_result == 2:
    print("Soil Health: GOOD")

elif soil_result == 1:
    print("Soil Health: MEDIUM")

else:
    print("Soil Health: POOR")


# Display crop yield

print(
    "Predicted Crop Yield:",
    round(yield_result, 2),
    "tons/hectare"
)