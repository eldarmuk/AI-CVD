# LEGACY V1: this is historical analysis, not the canonical primary pipeline.
"""Compatibility entrypoint for VAE AUPRC evaluation.

The implementation lives in src.pipelines.03_evaluate; this wrapper preserves
the documented 03b command name and supports `--mode vae`.
"""

from __future__ import annotations

if __name__ == "__main__":
    raise SystemExit("LEGACY EXPERIMENT: retired from the primary task. Use python -m src.ai_cvd.cli --help. Training migration is intentionally deferred.")

import importlib


def main() -> None:
    evaluator = importlib.import_module("src.pipelines.03_evaluate")
    evaluator.main()


if __name__ == "__main__":
    main()
