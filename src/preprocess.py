import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_COL = "churn"
ID_COL = "customer_id"

CATEGORICAL_COLS = [
    "gender",
    "region",
    "plan_type",
    "contract_type",
]

FEATURE_COLS = [
    "age",
    "gender",
    "region",
    "plan_type",
    "tenure_months",
    "data_used_gb",
    "call_minutes",
    "calls_made",
    "sms_count",
    "revenue_inr",
    "complaints_6m",
    "network_issues",
    "customer_service_calls",
    "payment_delay",
    "roaming_usage",
    "international_calls",
    "device_age_months",
    "satisfaction_score",
    "contract_type",
    "auto_payment",
]


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_and_impute(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    numeric_cols = df.select_dtypes(include="number").columns

    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    return df


def encode_features(df: pd.DataFrame):
    df = df.copy()

    df = pd.get_dummies(
        df,
        columns=CATEGORICAL_COLS,
        drop_first=False,
    )

    feature_cols = [
        col
        for col in df.columns
        if col not in [ID_COL, TARGET_COL]
    ]

    return df, feature_cols


def split_data(
    df: pd.DataFrame,
    feature_cols,
    test_size=0.2,
    random_state=42,
):
    X = df[feature_cols]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


def build_dataset(path: str):
    df = load_data(path)
    df = clean_and_impute(df)
    df, feature_cols = encode_features(df)

    X_train, X_test, y_train, y_test = split_data(
        df,
        feature_cols,
    )

    return X_train, X_test, y_train, y_test, feature_cols


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feature_cols = build_dataset(
        "data/telecom_master.csv"
    )

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")
    print(f"Churn rate (train): {y_train.mean():.3f}")
