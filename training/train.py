"""Train an income classifier and audit it for fairness across sex and race.

Usage:
    python -m training.train              # real data (UCI Adult, downloaded from OpenML)
    python -m training.train --synthetic  # small synthetic dataset (offline, used in CI)

Outputs (in models/):
    model.joblib          the trained scikit-learn pipeline
    fairness_report.json  overall and per-group metrics, gaps and bootstrap CIs
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from training.fairness import bootstrap_ci, fairness_gaps, group_metrics

NUMERIC = ["age", "education_num", "hours_per_week", "capital_gain", "capital_loss"]
CATEGORICAL = ["workclass", "marital_status", "occupation"]
FEATURES = NUMERIC + CATEGORICAL
SENSITIVE = ["sex", "race"]  # used ONLY for auditing, never as model inputs
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def load_adult() -> pd.DataFrame:
    """UCI Adult census dataset from OpenML (a standard fairness benchmark)."""
    from sklearn.datasets import fetch_openml

    data = fetch_openml("adult", version=2, as_frame=True)
    df = data.frame.rename(columns=lambda c: c.replace("-", "_"))
    df["label"] = (df["class"].astype(str).str.strip() == ">50K").astype(int)
    df = df[FEATURES + SENSITIVE + ["label"]].dropna()
    for col in CATEGORICAL + SENSITIVE:
        df[col] = df[col].astype(str)
    return df


def make_synthetic(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Small synthetic dataset with the same schema (for tests and offline runs)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(18, 70, n),
        "education_num": rng.integers(1, 17, n),
        "hours_per_week": rng.integers(10, 70, n),
        "capital_gain": rng.choice([0, 0, 0, 5000], n),
        "capital_loss": rng.choice([0, 0, 0, 1500], n),
        "workclass": rng.choice(["Private", "Self-emp", "Gov"], n),
        "marital_status": rng.choice(["Married", "Never-married", "Divorced"], n),
        "occupation": rng.choice(["Tech", "Sales", "Service", "Craft"], n),
        "sex": rng.choice(["Male", "Female"], n),
        "race": rng.choice(["White", "Black", "Other"], n),
    })
    score = 0.06 * df["education_num"] + 0.03 * df["hours_per_week"] + 0.02 * df["age"]
    df["label"] = (score + rng.normal(0, 0.5, n) > score.median()).astype(int)
    return df


def build_pipeline() -> Pipeline:
    preprocess = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([
        ("preprocess", preprocess),
        ("clf", LogisticRegression(max_iter=1000)),
    ])


def audit(y_true, y_pred, df_test: pd.DataFrame) -> dict:
    report = {}
    for attr in SENSITIVE:
        groups = df_test[attr].to_numpy()
        report[attr] = {
            "per_group": group_metrics(y_true, y_pred, groups),
            "gaps": fairness_gaps(y_true, y_pred, groups),
            "equalized_odds_ci": bootstrap_ci(y_true, y_pred, groups),
        }
    return report


def main(synthetic: bool = False, out_dir: Path = MODEL_DIR) -> dict:
    df = make_synthetic() if synthetic else load_adult()
    train_df, test_df = train_test_split(
        df, test_size=0.25, random_state=42, stratify=df["label"]
    )

    model = build_pipeline()
    model.fit(train_df[FEATURES], train_df["label"])

    proba = model.predict_proba(test_df[FEATURES])[:, 1]
    pred = (proba >= 0.5).astype(int)
    y_test = test_df["label"].to_numpy()

    report = {
        "trained_at": datetime.now(UTC).isoformat(),
        "dataset": "synthetic" if synthetic else "UCI Adult (OpenML v2)",
        "n_train": len(train_df),
        "n_test": len(test_df),
        "overall": {
            "accuracy": float(accuracy_score(y_test, pred)),
            "roc_auc": float(roc_auc_score(y_test, proba)),
        },
        "fairness": audit(y_test, pred, test_df),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "model.joblib")
    (out_dir / "fairness_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"overall": report["overall"],
                      "sex_gaps": report["fairness"]["sex"]["gaps"]}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true", help="use synthetic data")
    args = parser.parse_args()
    main(synthetic=args.synthetic)
