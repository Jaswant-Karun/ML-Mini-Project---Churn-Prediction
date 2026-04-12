from __future__ import annotations

import argparse
import json
import os
from typing import Dict

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from preprocess import build_preprocessor, load_and_clean_data, split_features_target


def train_and_select_model(data_path: str, model_output_path: str, metrics_output_path: str) -> Dict:
    df = load_and_clean_data(data_path)
    x, y = split_features_target(df)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(x_train)

    candidate_models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "DecisionTree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42),
    }

    results = {}
    best_name = None
    best_f1 = -1.0
    best_pipeline = None

    for name, estimator in candidate_models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )

        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results[name] = {
            "accuracy": round(float(acc), 4),
            "f1_score": round(float(f1), 4),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
        }

        if f1 > best_f1:
            best_f1 = f1
            best_name = name
            best_pipeline = pipeline

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(best_pipeline, model_output_path)

    summary = {
        "best_model": best_name,
        "results": results,
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
    }

    os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
    with open(metrics_output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train churn prediction models and save the best one.")
    parser.add_argument(
        "--data",
        default="data/Telco-Customer-Churn.csv",
        help="Path to Telco churn CSV file.",
    )
    parser.add_argument(
        "--model-out",
        default="model/churn_model.pkl",
        help="Path to save trained model pipeline.",
    )
    parser.add_argument(
        "--metrics-out",
        default="model/metrics.json",
        help="Path to save model comparison metrics.",
    )
    args = parser.parse_args()

    summary = train_and_select_model(args.data, args.model_out, args.metrics_out)
    print("Best model:", summary["best_model"])
    for model_name, values in summary["results"].items():
        print(f"{model_name}: accuracy={values['accuracy']}, f1={values['f1_score']}")


if __name__ == "__main__":
    main()
