import pandas as pd

# Load dataset
df = pd.read_csv("data/real_agriculture.csv")

print("================================")
print("SMART AGRICULTURE DATASET")
print("================================")

print("\nDataset shape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nDataset statistics:")
print(df.describe())