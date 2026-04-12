from __future__ import annotations

import argparse
import json

import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from preprocess import load_and_clean_data, split_features_target


def evaluate_model(model_path: str, data_path: str) -> dict:
    df = load_and_clean_data(data_path)
    x, y = split_features_target(df)

    _, x_test, _, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    model = joblib.load(model_path)
    y_pred = model.predict(x_test)

    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved churn prediction model.")
    parser.add_argument("--model", default="model/churn_model.pkl", help="Path to trained model.")
    parser.add_argument(
        "--data",
        default="data/Telco-Customer-Churn.csv",
        help="Path to Telco churn CSV file.",
    )
    args = parser.parse_args()

    metrics = evaluate_model(args.model, args.data)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
