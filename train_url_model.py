from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from phish_reimpl.url_analysis import extract_url_features


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "datasets" / "url" / "PhiUSIIL_Phishing_URL_Dataset.csv"
DEFAULT_MODEL_OUT = ROOT / "artifacts" / "url_rf_phiusiil.pkl"
DEFAULT_SUMMARY_OUT = ROOT / "artifacts" / "url_training_summary.json"

FEATURE_COLUMNS = [
    "url_length",
    "hostname_length",
    "path_length",
    "query_length",
    "tld_length",
    "num_subdomains",
    "num_digits",
    "digit_ratio",
    "num_letters",
    "letter_ratio",
    "num_special_chars",
    "special_char_ratio",
    "num_hyphens",
    "num_dots",
    "num_slashes",
    "num_at_signs",
    "num_equals",
    "num_question",
    "num_ampersand",
    "num_percent",
    "has_https",
    "has_ip_host",
    "has_punycode",
    "has_obfuscation",
    "keyword_hits",
    "hostname_hyphen_count",
    "path_token_count",
    "char_continuation_rate",
    "avg_token_length",
    "brand_like_terms",
]


def load_dataset(dataset_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(dataset_path, encoding="utf-8-sig")
    df["URL"] = df["URL"].astype(str)
    x = pd.DataFrame([extract_url_features(url) for url in df["URL"]])
    # PhiUSIIL uses 0=phishing, 1=benign. Convert to 1=phishing.
    y = (df["label"].astype(int) == 0).astype(int)
    return x, y


def build_models() -> dict[str, Pipeline]:
    prep = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                FEATURE_COLUMNS,
            )
        ]
    )
    return {
        "logreg": Pipeline(
            steps=[
                ("prep", prep),
                ("clf", LogisticRegression(max_iter=2000)),
            ]
        ),
        "rf": Pipeline(
            steps=[
                ("prep", prep),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=18,
                        min_samples_leaf=2,
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def summarize(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def train_and_select(dataset_path: Path) -> tuple[str, Pipeline, dict[str, dict[str, float]]]:
    x, y = load_dataset(dataset_path)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    best_name = ""
    best_model: Pipeline | None = None
    best_f1 = -1.0
    summary: dict[str, dict[str, float]] = {}
    for name, model in build_models().items():
        model.fit(x_train, y_train)
        preds = model.predict(x_test)
        metrics = summarize(y_test, preds)
        summary[name] = metrics
        if metrics["f1"] > best_f1:
            best_name = name
            best_model = model
            best_f1 = metrics["f1"]

    if best_model is None:
        raise RuntimeError("No URL model trained successfully.")
    return best_name, best_model, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_OUT)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY_OUT)
    args = parser.parse_args()

    best_name, best_model, summary = train_and_select(args.dataset)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)

    with args.model_out.open("wb") as handle:
        pickle.dump(best_model, handle)
    with args.summary_out.open("w") as handle:
        json.dump({"best_model": best_name, "metrics": summary}, handle, indent=2)

    print(f"best_model: {best_name}")
    print(f"model_out: {args.model_out}")
    print(f"summary_out: {args.summary_out}")


if __name__ == "__main__":
    main()
