import json
import os

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal


MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

MODEL_NAME = "telecom-churn-model"
MODEL_ALIAS = "production"
MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

app = FastAPI(title="Telecom Churn Prediction API")

model = mlflow.sklearn.load_model(MODEL_URI)

_client = mlflow.MlflowClient(
    tracking_uri=MLFLOW_TRACKING_URI
)

_mv = _client.get_model_version_by_alias(
    MODEL_NAME,
    MODEL_ALIAS,
)

_artifact_path = _client.download_artifacts(
    _mv.run_id,
    "feature_columns.json",
)

with open(_artifact_path) as f:
    FEATURE_COLUMNS = json.load(f)


class Customer(BaseModel):
    age: float
    gender: Literal["Female", "Male"]
    region: Literal[
        "Bangalore",
        "Chennai",
        "Delhi",
        "Kolkata",
        "Mumbai",
    ]
    plan_type: Literal["Postpaid", "Prepaid"]
    tenure_months: int
    data_used_gb: float
    call_minutes: int
    calls_made: int
    sms_count: int
    revenue_inr: float
    complaints_6m: int
    network_issues: int
    customer_service_calls: int
    payment_delay: int
    roaming_usage: int
    international_calls: int
    device_age_months: int
    satisfaction_score: int
    contract_type: Literal[
        "Annual",
        "Monthly",
        "Two-Year",
    ]
    auto_payment: int = Field(ge=0, le=1)


def build_feature_row(customer: Customer) -> pd.DataFrame:
    raw = customer.model_dump()

    row = {col: 0 for col in FEATURE_COLUMNS}

    numeric_columns = [
        "age",
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
        "auto_payment",
    ]

    for col in numeric_columns:
        row[col] = raw[col]

    for prefix, value in [
        ("gender", raw["gender"]),
        ("region", raw["region"]),
        ("plan_type", raw["plan_type"]),
        ("contract_type", raw["contract_type"]),
    ]:
        one_hot_col = f"{prefix}_{value}"

        if one_hot_col in row:
            row[one_hot_col] = 1

    return pd.DataFrame(
        [row],
        columns=FEATURE_COLUMNS,
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_uri": MODEL_URI,
        "model_version": _mv.version,
    }


@app.post("/predict")
def predict(customer: Customer):
    row = build_feature_row(customer)

    prediction = model.predict(row)[0]
    probability = model.predict_proba(row)[0][1]

    return {
        "churn_prediction": int(prediction),
        "churn_probability": round(
            float(probability),
            4,
        ),
    }
