"""Portable tabular modelling core for the AutoSlide repository."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC


FEATURES = [
    "DEM",
    "slope",
    "aspect",
    "plan",
    "profile",
    "rainfall",
    "NDVI",
    "land_use",
    "road",
    "river",
    "temyear",
    "roughness",
]
LABEL = "landslide"


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load and validate a tabular landslide dataset."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    elif path.suffix.lower() in {".xls", ".xlsx"}:
        frame = pd.read_excel(path)
    else:
        raise ValueError("Input must be CSV, XLS, or XLSX")

    missing = [column for column in FEATURES + [LABEL] if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if frame[FEATURES].isna().any().any():
        raise ValueError("Predictor columns contain missing values")
    if not np.isfinite(frame[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("Predictor columns contain non-finite values")
    labels = set(frame[LABEL].unique())
    if not labels.issubset({0, 1}):
        raise ValueError("The landslide column must contain only 0 and 1")
    if len(labels) < 2:
        raise ValueError("Both label classes are required")
    return frame.copy()


def make_model(name: str, seed: int):
    """Create one supported classifier with deterministic defaults."""
    name = name.lower()
    if name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=1,
        )
    if name == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            iterations=200,
            depth=5,
            learning_rate=0.05,
            verbose=False,
            random_seed=seed,
            thread_count=1,
        )
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1)
    if name == "logistic_regression":
        return Pipeline([
            ("scale", MinMaxScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=seed)),
        ])
    if name == "mlp":
        return Pipeline([
            ("scale", MinMaxScaler()),
            ("model", MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=seed)),
        ])
    if name == "svm":
        return Pipeline([
            ("scale", MinMaxScaler()),
            ("model", SVC(probability=True, random_state=seed)),
        ])
    raise ValueError(f"Unsupported model: {name}")


def run_benchmark(frame: pd.DataFrame, model_name: str, rounds: int = 1,
                  test_size: float = 0.2, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run repeated stratified evaluation and return metrics and predictions."""
    X = frame[FEATURES].astype(float)
    y = frame[LABEL].astype(int)
    metric_rows = []
    prediction_rows = []
    for round_index in range(1, rounds + 1):
        round_seed = seed + round_index - 1
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=round_seed
        )
        model = make_model(model_name, round_seed)
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        probability = model.predict_proba(X_test)[:, 1]
        metric_rows.append({
            "round": round_index,
            "accuracy": accuracy_score(y_test, predicted),
            "precision": precision_score(y_test, predicted, zero_division=0),
            "recall": recall_score(y_test, predicted, zero_division=0),
            "f1": f1_score(y_test, predicted, zero_division=0),
            "auc": roc_auc_score(y_test, probability),
        })
        prediction_rows.append(pd.DataFrame({
            "round": round_index,
            "row_index": y_test.index,
            "observed": y_test.to_numpy(),
            "predicted": predicted,
            "probability": probability,
        }))
    return pd.DataFrame(metric_rows), pd.concat(prediction_rows, ignore_index=True)


def save_results(metrics: pd.DataFrame, predictions: pd.DataFrame, output: str | Path) -> dict:
    """Write tabular outputs and return the summary dictionary."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "metrics.csv", index=False)
    predictions.to_csv(output / "predictions.csv", index=False)
    numeric = metrics.drop(columns=["round"]).agg(["mean", "std"]).round(6)
    summary = {row: {column: float(numeric.loc[row, column]) for column in numeric.columns}
               for row in numeric.index}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run portable AutoSlide model evaluation")
    parser.add_argument("--input", required=True, help="CSV, XLS, or XLSX input dataset")
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument("--model", default="random_forest", choices=[
        "xgboost", "catboost", "random_forest", "logistic_regression", "mlp", "svm"
    ])
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.rounds < 1 or not 0 < args.test_size < 1:
        raise ValueError("rounds must be positive and test-size must be between 0 and 1")
    frame = load_dataset(args.input)
    metrics, predictions = run_benchmark(frame, args.model, args.rounds, args.test_size, args.seed)
    summary = save_results(metrics, predictions, args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

