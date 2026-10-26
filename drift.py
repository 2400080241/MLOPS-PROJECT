import pandas as pd
import numpy as np
from evidently import Report
from evidently.presets import DataDriftPreset


# ============================================================
# SMART AGRICULTURE - DATA DRIFT DETECTION
# ============================================================

print("=" * 50)
print("SMART AGRICULTURE DATA DRIFT DETECTION")
print("=" * 50)


# ------------------------------------------------------------
# 1. Load real Kaggle dataset
# ------------------------------------------------------------

DATA_PATH = "data/real_agriculture.csv"

df = pd.read_csv(DATA_PATH)

print("\nOriginal dataset shape:")
print(df.shape)


# ------------------------------------------------------------
# 2. Select the features used by our ML model
# ------------------------------------------------------------

features = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
    "soil_moisture"
]

df = df[features].copy()

print("\nSelected features:")
print(features)

print("\nSelected dataset shape:")
print(df.shape)


# ------------------------------------------------------------
# 3. Create reference and current datasets
# ------------------------------------------------------------

# Reference = baseline data
reference_data = df.iloc[:1100].copy()

# Current = data that represents new incoming data
current_data = df.iloc[1100:].copy()


print("\nReference data shape:")
print(reference_data.shape)

print("\nCurrent data shape:")
print(current_data.shape)


# ------------------------------------------------------------
# 4. Simulate realistic data drift
# ------------------------------------------------------------

# IMPORTANT:
# This is only for demonstrating how an MLOps
# monitoring system detects drift.

current_data = current_data.copy()

current_data["temperature"] = current_data["temperature"] + 5

current_data["humidity"] = (
    current_data["humidity"] * 0.85
).clip(0, 100)

current_data["soil_moisture"] = (
    current_data["soil_moisture"] * 0.75
).clip(0, 100)

current_data["rainfall"] = (
    current_data["rainfall"] * 0.60
).clip(lower=0)


# ------------------------------------------------------------
# 5. Save reference and current datasets
# ------------------------------------------------------------

reference_data.to_csv(
    "data/reference_data.csv",
    index=False
)

current_data.to_csv(
    "data/current_data.csv",
    index=False
)

print("\nReference and current datasets saved.")


# ------------------------------------------------------------
# 6. Run Evidently Data Drift Report
# ------------------------------------------------------------

print("\nRunning Evidently drift detection...")

report = Report(
    metrics=[
        DataDriftPreset()
    ]
)

snapshot = report.run(
    reference_data=reference_data,
    current_data=current_data
)


# ------------------------------------------------------------
# 7. Save HTML report
# ------------------------------------------------------------

snapshot.save_html("drift_report.html")

print("\n" + "=" * 50)
print("DATA DRIFT REPORT GENERATED")
print("=" * 50)

print("\nReport:")
print("drift_report.html")

print("\nOpen the report in your browser.")