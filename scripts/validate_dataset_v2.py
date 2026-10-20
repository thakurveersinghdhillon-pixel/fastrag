#!/usr/bin/env python3
"""Command-line script to validate FastRAG Research Evaluation Dataset v2.

Exits with code 0 if valid, code 1 if validation fails.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.dataset_validation_v2 import validate_dataset_v2


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate FastRAG Research Evaluation Dataset v2 integrity, schemas, and split isolation."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "v2",
        help="Path to directory containing dataset v2 files (default: datasets/v2/)",
    )
    args = parser.parse_args()

    print(f"Validating Dataset v2 at: {args.dataset_dir}")
    result = validate_dataset_v2(args.dataset_dir)
    print(result.summary())

    if result.stats.get("categories"):
        print("\nCategory Distribution:")
        for cat, cnt in sorted(result.stats["categories"].items()):
            print(f"  - {cat}: {cnt}")

    if result.stats.get("provenance"):
        print("\nProvenance Status Distribution:")
        for prov, cnt in sorted(result.stats["provenance"].items()):
            print(f"  - {prov}: {cnt}")

    if result.is_valid:
        print("\nAll Dataset v2 validation checks passed successfully.")
        return 0
    else:
        print(f"\nDataset v2 validation failed with {len(result.errors)} error(s).", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
