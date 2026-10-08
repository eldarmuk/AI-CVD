"""Connect processed telemetry to the retained Study B feature/model interfaces."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ai_cvd.config import public_configuration
from ai_cvd.data.synthetic import COLUMNS
from ai_cvd.features.episode import CHANNELS, STEP_NS, episode_statistics
from ai_cvd.features.temporal import slot_starts, temporal_record

from .runner import safe_artifact, verify
from .storage import connect


def model_fixture(run):
    """Only fictional evaluation; processing itself accepts authorized canonical exports."""
    run = Path(run)
    meta = verify(run)
    if not meta["synthetic"]:
        raise ValueError("Public model demonstration accepts fictional pipeline runs only")
    episodes = json.loads((run / "episodes.json").read_text())
    config = dict(public_configuration(), _canonical_feature_names=list(COLUMNS))
    rows, records, grids = [], {}, {}
    with connect(run / "telemetry.duckdb", read_only=True) as db:
        for patient in json.loads((run / "patients.json").read_text()):
            frame = db.read_parquet(str(safe_artifact(run, patient["features"]))).df()
            grid = frame[list(COLUMNS)].to_numpy(dtype=float)
            start = pd.Timestamp(frame.iloc[0].timestamp).value
            starts = start + np.arange(len(grid), dtype=np.int64) * STEP_NS
            # Preserve actual last-valid-contributor recency from the retained causal buckets.
            contributors = {}
            for channel in CHANNELS:
                valid = frame["observed_" + channel].astype(bool).to_numpy()
                recency = frame["time_since_last_" + channel].to_numpy(dtype=float)
                contributors[channel] = (
                    starts[valid] + STEP_NS - np.rint(recency[valid] * 60e9).astype(np.int64)
                ).tolist()
            for episode in episodes:
                if episode["patient"] != patient["subject_id"]:
                    continue
                anchor = pd.Timestamp(episode["anchor"]).value
                if anchor - 8 * 86400_000_000_000 < start or anchor > starts[-1] + STEP_NS:
                    raise ValueError(
                        "Study B example requires the full eight-day pre-alarm coverage"
                    )
                row = episode_statistics(
                    episode["patient"], episode, grid, start, contributors, config
                )
                offsets = (slot_starts(episode["anchor"]) - start) // STEP_NS
                records[row["episode_id"]] = temporal_record(row, grid[offsets, 6])
                grids[row["episode_id"]] = grid
                rows.append(row)
    alerts = [
        dict(
            synthetic=True,
            synthetic_subject=e["patient"],
            synthetic_alert_id=m["event_id"],
            timestamp=m["timestamp"],
            synthetic_outcome_code=m["severity"],
        )
        for e in episodes
        for m in e["constituents"]
    ]
    return alerts, episodes, rows, records, grids
