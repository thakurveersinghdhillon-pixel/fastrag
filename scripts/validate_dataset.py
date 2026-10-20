#!/usr/bin/env python3
"""Command-line script to validate FastRAG evaluation dataset.

Exits with code 0 if valid, code 1 if validation fails.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.dataset_validation import validate_dataset
from evaluation.dataset_validation_v2 import validate_dataset_v2


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate FastRAG evaluation dataset integrity and schema.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=PROJECT_ROOT / "datasets",
        help="Path to directory containing dataset files (default: datasets/)",
    )
    args = parser.parse_args()

    # Auto-detect whether target directory is dataset v1 or v2
    manifest_path = args.dataset_dir / "dataset_manifest.json"
    is_v2 = False
    if manifest_path.is_file():
        try:
            import json
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
                if str(manifest_data.get("version", "")).startswith("2."):
                    is_v2 = True
        except Exception:
            pass
    if not is_v2 and (args.dataset_dir / "queries_dev.jsonl").is_file():
        is_v2 = True

    print(f"Validating dataset at: {args.dataset_dir} (detected format: {'v2' if is_v2 else 'v1'})")
    result = validate_dataset_v2(args.dataset_dir) if is_v2 else validate_dataset(args.dataset_dir)
    print(result.summary())

    if result.is_valid:
        print("\nAll dataset validation checks passed successfully.")
        return 0
    else:
        print(f"\nDataset validation failed with {len(result.errors)} error(s).", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

