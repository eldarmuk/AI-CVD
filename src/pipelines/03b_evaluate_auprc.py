"""Compatibility entrypoint for VAE AUPRC evaluation.

The implementation lives in src.pipelines.03_evaluate; this wrapper preserves
the documented 03b command name and supports `--mode vae`.
"""

from __future__ import annotations

import importlib


def main() -> None:
    evaluator = importlib.import_module("src.pipelines.03_evaluate")
    evaluator.main()


if __name__ == "__main__":
    main()
