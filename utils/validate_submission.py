"""Local schema validator. This is not a competition-supplied validator."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_loader import read_source
from src.submission import validate_submission
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matching", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--check-ids", action="store_true", default=False, help="Check ID membership against test sources")
    args = parser.parse_args()
    try:
        root = Path(args.test_dir)
        frames = [read_source(root / f"test_source{i}.tsv", i) for i in (1, 2, 3)]
        report = validate_submission(args.matching, args.candidate, frames[0], pd.concat(frames[1:], ignore_index=True))
    except (ValueError, OSError) as exc:
        parser.exit(1, f"INVALID: {exc}\n")
    print(f"VALID (local schema checks): {report}")


if __name__ == "__main__":
    main()
