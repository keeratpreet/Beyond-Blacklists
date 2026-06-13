from __future__ import annotations

import argparse
from pathlib import Path

from phish_reimpl.visual import VisualAnalyzer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "cache",
    )
    parser.add_argument(
        "--force-raw-rebuild",
        action="store_true",
        help="Ignore bootstrap arrays and rebuild embeddings from raw logo files.",
    )
    args = parser.parse_args()

    cache_file = args.cache_dir / "reference_embeddings.pkl"
    if args.force_raw_rebuild and cache_file.exists():
        cache_file.unlink()

    analyzer = VisualAnalyzer(cache_dir=args.cache_dir)
    print(f"reference_entries: {len(analyzer.reference_entries)}")
    print(f"cache_file: {cache_file}")


if __name__ == "__main__":
    main()
