from __future__ import annotations

import csv
from pathlib import Path


def load_expected(dataset_dir: Path) -> dict[str, int]:
    expected = {}
    with (dataset_dir / "expected_labels.csv").open() as f:
        for row in csv.DictReader(f):
            expected[row["case_id"]] = int(row["expected_phish"])
    return expected


def score_results(results_file: Path, expected: dict[str, int]) -> dict[str, float | int]:
    tp = fp = tn = fn = 0
    with results_file.open() as f:
        for line_number, line in enumerate(f):
            parts = line.rstrip("\n").split("\t")
            if line_number == 0 and parts and parts[0] == "case_id":
                continue
            if len(parts) < 3 or parts[0] not in expected:
                continue
            pred = int(parts[2])
            y_true = expected[parts[0]]
            if pred == 1 and y_true == 1:
                tp += 1
            elif pred == 1 and y_true == 0:
                fp += 1
            elif pred == 0 and y_true == 0:
                tn += 1
            else:
                fn += 1

    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    accuracy = (tp + tn) / total if total else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "total": total,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
