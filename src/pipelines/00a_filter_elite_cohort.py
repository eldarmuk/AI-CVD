"""Canonical v2 entry point. Historical implementation is in src/archive/legacy_v1.
This command now builds one versioned feature/episode/manifest run; use --help.
"""
from pathlib import Path
import sys
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.ai_cvd.cli import main

if __name__ == "__main__":
    main()
