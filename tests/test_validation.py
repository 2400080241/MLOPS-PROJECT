
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "real_agriculture.csv"


def test_dataset_exists():
    assert DATA_PATH.exists(), f"Dataset not found: {DATA_PATH}"


def test_dataset_is_not_empty():
    df = pd.read_csv(DATA_PATH)
    assert not df.empty


def test_dataset_has_required_columns():
    df = pd.read_csv(DATA_PATH)

    required_columns = [
        "N", "P", "K", "temperature",
        "humidity", "ph", "rainfall", "label",
    ]

    assert all(column in df.columns for column in required_columns)


def test_dataset_has_no_missing_values():
    df = pd.read_csv(DATA_PATH)
    assert not df.isnull().any().any()


def test_dataset_has_no_duplicate_rows():
    df = pd.read_csv(DATA_PATH)
    assert not df.duplicated().any()
