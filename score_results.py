from __future__ import annotations

import argparse
import csv
from pathlib import Path

from phish_reimpl.scoring import score_results


def load_expected_labels(dataset_dir: Path) -> dict[str, int]:
    expected_file = dataset_dir / "expected_labels.csv"
    if expected_file.exists():
        with expected_file.open() as handle:
            return {row["case_id"]: int(row["expected_phish"]) for row in csv.DictReader(handle)}

    # Fallback for tiny ad hoc demo folders without an explicit labels file.
    labels: dict[str, int] = {}
    for case_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
        labels[case_dir.name] = int(case_dir.name.endswith("_phish"))
    return labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--results-file", type=Path, required=True)
    args = parser.parse_args()

    metrics = score_results(args.results_file, load_expected_labels(args.dataset_dir))
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
