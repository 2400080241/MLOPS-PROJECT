import pandas as pd
import os

# Load Kaggle dataset
input_file = "data/real_agriculture.csv"

df = pd.read_csv(input_file)

# Select the features we need
clean_df = df[
    [
        "N",
        "P",
        "K",
        "temperature",
        "humidity",
        "ph",
        "rainfall",
        "soil_moisture",
        "label"
    ]
].copy()

# Rename columns to make them easier to use
clean_df.rename(
    columns={
        "N": "nitrogen",
        "P": "phosphorus",
        "K": "potassium",
        "ph": "soil_ph"
    },
    inplace=True
)

# Create processed-data directory
os.makedirs("data/processed", exist_ok=True)

# Save cleaned dataset
output_file = "data/processed/agriculture_clean.csv"

clean_df.to_csv(output_file, index=False)

print("Clean dataset created successfully!")
print("File:", output_file)
print("Shape:", clean_df.shape)

print("\nColumns:")
print(clean_df.columns.tolist())

print("\nFirst 5 rows:")
print(clean_df.head())

print("\nCrop labels:")
print(clean_df["label"].value_counts())