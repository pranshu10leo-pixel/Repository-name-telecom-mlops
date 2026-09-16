from pathlib import Path

from src.preprocess import load_data, clean_and_impute, FEATURE_COLS


DATA_PATH = Path("data/telecom_master.csv")


def test_load_data():
    df = load_data(DATA_PATH)
    assert df.shape[0] == 3000
    assert "churn" in df.columns


def test_feature_columns():
    df = load_data(DATA_PATH)
    for col in FEATURE_COLS:
        assert col in df.columns


def test_clean_and_impute():
    df = load_data(DATA_PATH)
    cleaned = clean_and_impute(df)
    assert cleaned.isnull().sum().sum() == 0
