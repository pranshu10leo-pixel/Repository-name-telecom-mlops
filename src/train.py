import json
import os

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from src.preprocess import build_dataset


MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "telecom-churn"


def train_model(model, model_name, X_train, X_test, y_train, y_test, feature_cols):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=model_name):

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)

        mlflow.log_param("model_type", model_name)
        mlflow.log_param("n_features", len(feature_cols))

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        os.makedirs("artifacts", exist_ok=True)

        with open("artifacts/feature_columns.json", "w") as f:
            json.dump(feature_cols, f, indent=2)

        mlflow.log_artifact("artifacts/feature_columns.json")

        if model_name == "Random Forest":
            mlflow.sklearn.log_model(
                model,
                "model",
                skops_trusted_types=[
                    "sklearn.tree._tree.Tree"
                ],
                registered_model_name="telecom-churn-model",
            )
        else:
            mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name="telecom-churn-model",
            )

        print(
            f"{model_name}: "
            f"accuracy={accuracy:.4f}, "
            f"precision={precision:.4f}, "
            f"recall={recall:.4f}, "
            f"f1={f1:.4f}, "
            f"roc_auc={roc_auc:.4f}"
        )


def main():
    X_train, X_test, y_train, y_test, feature_cols = build_dataset(
        "data/telecom_master.csv"
    )

    train_model(
        make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
        "Logistic Regression",
        X_train,
        X_test,
        y_train,
        y_test,
        feature_cols,
    )

    train_model(
        RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest",
        X_train,
        X_test,
        y_train,
        y_test,
        feature_cols,
    )


if __name__ == "__main__":
    main()
