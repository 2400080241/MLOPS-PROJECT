import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# MLFLOW SETUP
# ============================================================

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Smart Agriculture Prediction")


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv("data/agriculture.csv")

print("\n========================================")
print("SMART AGRICULTURE ML SYSTEM")
print("========================================")

print("\nDataset shape:")
print(df.shape)


# ============================================================
# DATA VALIDATION
# ============================================================

print("\n========================================")
print("DATA VALIDATION")
print("========================================")

print("\nMissing values:")
print(df.isnull().sum())

print("\nTotal missing values:")
print(df.isnull().sum().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "temperature",
    "humidity",
    "soil_moisture",
    "rainfall",
    "soil_ph",
    "nitrogen",
    "phosphorus",
    "potassium",
    "crop_yield",
    "soil_health",
    "irrigation_required"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    print("\nERROR: Missing columns:")
    print(missing_columns)

    raise ValueError(
        "Dataset does not contain all required columns."
    )


print("\nAll required columns are present.")


# ============================================================
# INPUT FEATURES
# ============================================================

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


# ============================================================
# GENERAL MODEL SETTINGS
# ============================================================

N_ESTIMATORS = 100
RANDOM_STATE = 42
TEST_SIZE = 0.20

TRUSTED_TYPES = [
    "sklearn.tree._tree.Tree"
]


# ============================================================
# MODEL 1: IRRIGATION
# ============================================================

print("\n")
print("========================================")
print("IRRIGATION PREDICTION")
print("========================================")


y_irrigation = df["irrigation_required"]


X_train_irrigation, X_test_irrigation, \
y_train_irrigation, y_test_irrigation = train_test_split(

    X,
    y_irrigation,

    test_size=TEST_SIZE,

    random_state=RANDOM_STATE,

    stratify=y_irrigation
)


with mlflow.start_run(
    run_name="Irrigation_Model"
):

    # --------------------------------------------------------
    # CREATE MODEL
    # --------------------------------------------------------

    irrigation_model = RandomForestClassifier(

        n_estimators=N_ESTIMATORS,

        random_state=RANDOM_STATE
    )


    # --------------------------------------------------------
    # TRAIN MODEL
    # --------------------------------------------------------

    irrigation_model.fit(

        X_train_irrigation,

        y_train_irrigation
    )


    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    irrigation_prediction = irrigation_model.predict(

        X_test_irrigation
    )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    irrigation_accuracy = accuracy_score(

        y_test_irrigation,

        irrigation_prediction
    )


    irrigation_precision = precision_score(

        y_test_irrigation,

        irrigation_prediction,

        zero_division=0
    )


    irrigation_recall = recall_score(

        y_test_irrigation,

        irrigation_prediction,

        zero_division=0
    )


    irrigation_f1 = f1_score(

        y_test_irrigation,

        irrigation_prediction,

        zero_division=0
    )


    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print(
        "Accuracy:",
        round(irrigation_accuracy, 4)
    )

    print(
        "Precision:",
        round(irrigation_precision, 4)
    )

    print(
        "Recall:",
        round(irrigation_recall, 4)
    )

    print(
        "F1 Score:",
        round(irrigation_f1, 4)
    )


    print("\nClassification Report:")

    print(
        classification_report(

            y_test_irrigation,

            irrigation_prediction,

            zero_division=0
        )
    )


    # --------------------------------------------------------
    # MLFLOW PARAMETERS
    # --------------------------------------------------------

    mlflow.log_param(
        "model_type",
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
        "dataset",
        "data/agriculture.csv"
    )

    mlflow.log_param(
        "dataset_rows",
        len(df)
    )

    mlflow.log_param(
        "dataset_columns",
        len(df.columns)
    )

    mlflow.log_param(
        "features",
        ",".join(features)
    )


    # --------------------------------------------------------
    # MLFLOW METRICS
    # --------------------------------------------------------

    mlflow.log_metric(
        "accuracy",
        irrigation_accuracy
    )

    mlflow.log_metric(
        "precision",
        irrigation_precision
    )

    mlflow.log_metric(
        "recall",
        irrigation_recall
    )

    mlflow.log_metric(
        "f1_score",
        irrigation_f1
    )


    # --------------------------------------------------------
    # SAVE MODEL TO MLFLOW
    # --------------------------------------------------------

    mlflow.sklearn.log_model(

        irrigation_model,

        name="irrigation_model",

        skops_trusted_types=TRUSTED_TYPES
    )


# ============================================================
# MODEL 2: SOIL HEALTH
# ============================================================

print("\n")
print("========================================")
print("SOIL HEALTH PREDICTION")
print("========================================")


y_soil = df["soil_health"]


X_train_soil, X_test_soil, \
y_train_soil, y_test_soil = train_test_split(

    X,
    y_soil,

    test_size=TEST_SIZE,

    random_state=RANDOM_STATE,

    stratify=y_soil
)


with mlflow.start_run(
    run_name="Soil_Health_Model"
):

    # --------------------------------------------------------
    # CREATE MODEL
    # --------------------------------------------------------

    soil_model = RandomForestClassifier(

        n_estimators=N_ESTIMATORS,

        random_state=RANDOM_STATE
    )


    # --------------------------------------------------------
    # TRAIN MODEL
    # --------------------------------------------------------

    soil_model.fit(

        X_train_soil,

        y_train_soil
    )


    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    soil_prediction = soil_model.predict(

        X_test_soil
    )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    soil_accuracy = accuracy_score(

        y_test_soil,

        soil_prediction
    )


    soil_precision = precision_score(

        y_test_soil,

        soil_prediction,

        average="weighted",

        zero_division=0
    )


    soil_recall = recall_score(

        y_test_soil,

        soil_prediction,

        average="weighted",

        zero_division=0
    )


    soil_f1 = f1_score(

        y_test_soil,

        soil_prediction,

        average="weighted",

        zero_division=0
    )


    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print(
        "Accuracy:",
        round(soil_accuracy, 4)
    )

    print(
        "Precision:",
        round(soil_precision, 4)
    )

    print(
        "Recall:",
        round(soil_recall, 4)
    )

    print(
        "F1 Score:",
        round(soil_f1, 4)
    )


    print("\nClassification Report:")

    print(
        classification_report(

            y_test_soil,

            soil_prediction,

            zero_division=0
        )
    )


    # --------------------------------------------------------
    # MLFLOW PARAMETERS
    # --------------------------------------------------------

    mlflow.log_param(
        "model_type",
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
        "dataset",
        "data/agriculture.csv"
    )

    mlflow.log_param(
        "dataset_rows",
        len(df)
    )

    mlflow.log_param(
        "dataset_columns",
        len(df.columns)
    )

    mlflow.log_param(
        "features",
        ",".join(features)
    )


    # --------------------------------------------------------
    # MLFLOW METRICS
    # --------------------------------------------------------

    mlflow.log_metric(
        "accuracy",
        soil_accuracy
    )

    mlflow.log_metric(
        "precision",
        soil_precision
    )

    mlflow.log_metric(
        "recall",
        soil_recall
    )

    mlflow.log_metric(
        "f1_score",
        soil_f1
    )


    # --------------------------------------------------------
    # SAVE MODEL TO MLFLOW
    # --------------------------------------------------------

    mlflow.sklearn.log_model(

        soil_model,

        name="soil_health_model",

        skops_trusted_types=TRUSTED_TYPES
    )


# ============================================================
# MODEL 3: CROP YIELD
# ============================================================

print("\n")
print("========================================")
print("CROP YIELD PREDICTION")
print("========================================")


y_yield = df["crop_yield"]


X_train_yield, X_test_yield, \
y_train_yield, y_test_yield = train_test_split(

    X,
    y_yield,

    test_size=TEST_SIZE,

    random_state=RANDOM_STATE
)


with mlflow.start_run(
    run_name="Crop_Yield_Model"
):

    # --------------------------------------------------------
    # CREATE MODEL
    # --------------------------------------------------------

    yield_model = RandomForestRegressor(

        n_estimators=N_ESTIMATORS,

        random_state=RANDOM_STATE
    )


    # --------------------------------------------------------
    # TRAIN MODEL
    # --------------------------------------------------------

    yield_model.fit(

        X_train_yield,

        y_train_yield
    )


    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    yield_prediction = yield_model.predict(

        X_test_yield
    )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    mae = mean_absolute_error(

        y_test_yield,

        yield_prediction
    )


    rmse = np.sqrt(

        mean_squared_error(

            y_test_yield,

            yield_prediction
        )
    )


    r2 = r2_score(

        y_test_yield,

        yield_prediction
    )


    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print(
        "MAE:",
        round(mae, 4)
    )

    print(
        "RMSE:",
        round(rmse, 4)
    )

    print(
        "R²:",
        round(r2, 4)
    )


    # --------------------------------------------------------
    # MLFLOW PARAMETERS
    # --------------------------------------------------------

    mlflow.log_param(
        "model_type",
        "RandomForestRegressor"
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
        "dataset",
        "data/agriculture.csv"
    )

    mlflow.log_param(
        "dataset_rows",
        len(df)
    )

    mlflow.log_param(
        "dataset_columns",
        len(df.columns)
    )

    mlflow.log_param(
        "features",
        ",".join(features)
    )


    # --------------------------------------------------------
    # MLFLOW METRICS
    # --------------------------------------------------------

    mlflow.log_metric(
        "MAE",
        mae
    )

    mlflow.log_metric(
        "RMSE",
        rmse
    )

    mlflow.log_metric(
        "R2",
        r2
    )


    # --------------------------------------------------------
    # SAVE MODEL TO MLFLOW
    # --------------------------------------------------------

    mlflow.sklearn.log_model(

        yield_model,

        name="crop_yield_model",

        skops_trusted_types=TRUSTED_TYPES
    )


    # --------------------------------------------------------
    # ACTUAL VS PREDICTED
    # --------------------------------------------------------

    print("\n")
    print("========================================")
    print("CROP YIELD - ACTUAL VS PREDICTED")
    print("========================================")


    for actual, predicted in zip(

        y_test_yield,

        yield_prediction

    ):

        print(

            "Actual:",

            round(actual, 2),

            "| Predicted:",

            round(predicted, 2)
        )


# ============================================================
# NEW FARM PREDICTION
# ============================================================

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


# ============================================================
# PREDICT IRRIGATION
# ============================================================

irrigation_result = irrigation_model.predict(

    new_farm

)[0]


# ============================================================
# PREDICT SOIL HEALTH
# ============================================================

soil_result = soil_model.predict(

    new_farm

)[0]


# ============================================================
# PREDICT CROP YIELD
# ============================================================

yield_result = yield_model.predict(

    new_farm

)[0]


# ============================================================
# DISPLAY IRRIGATION
# ============================================================

if irrigation_result == 1:

    print("Irrigation: REQUIRED")

else:

    print("Irrigation: NOT REQUIRED")


# ============================================================
# DISPLAY SOIL HEALTH
# ============================================================

if soil_result == 2:

    print("Soil Health: GOOD")

elif soil_result == 1:

    print("Soil Health: MEDIUM")

else:

    print("Soil Health: POOR")


# ============================================================
# DISPLAY CROP YIELD
# ============================================================

print(

    "Predicted Crop Yield:",

    round(yield_result, 2),

    "tons/hectare"

)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n")
print("========================================")
print("TRAINING COMPLETED SUCCESSFULLY")
print("========================================")

print("\nMLflow experiment:")
print("Smart Agriculture Prediction")

print("\nModels logged:")
print("1. Irrigation Model")
print("2. Soil Health Model")
print("3. Crop Yield Model")