"""Run invented arithmetic examples, never patient data or empirical model fitting."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hr_change.reproducibility import DEFAULT_CONFIG, run_synthetic


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="New directory outside the study source tree"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    try:
        manifest = run_synthetic(args.output, args.config)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Synthetic verification failed: {error}\n")
    print(
        json.dumps(
            {
                "synthetic_only": True,
                "verified": True,
                "artifact_sha256": manifest["artifacts_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
