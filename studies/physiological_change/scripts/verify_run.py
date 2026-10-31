"""Verify artifact hashes, source snapshot and exact synthetic semantic replay."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hr_change.reproducibility import verify_run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", dest="run_directory", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = verify_run(args.run_directory)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Run verification failed: {error}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
