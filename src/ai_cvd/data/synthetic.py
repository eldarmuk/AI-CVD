"""Arithmetic and seeded-noise fixtures: no source clinical records are read."""

import numpy as np
import pandas as pd

from ai_cvd.data.episodes import SYNTHETIC_PREFIX, group_episodes
from ai_cvd.features.episode import CHANNELS, STEP_NS, episode_statistics
from ai_cvd.features.temporal import slot_starts, temporal_record

COLUMNS = (*CHANNELS, "steps", "steps_delta_interval_minutes", "observed_steps_source")


def generate_alerts(patients=12):
    if patients < 6:
        raise ValueError("At least six subjects required")
    origin = pd.Timestamp("2099-01-15T12:00:00Z")
    alerts = []
    for i in range(patients):
        for visit in range(2):
            anchor = origin + pd.Timedelta(days=10 * visit, hours=i)
            for member in range(2):
                alerts.append(
                    {
                        "synthetic": True,
                        "synthetic_subject": f"{SYNTHETIC_PREFIX}{i:03d}",
                        "synthetic_alert_id": f"SYNTHETIC_ALERT_{i:03d}_{visit}_{member}",
                        "timestamp": (anchor + pd.Timedelta(minutes=5 * member)).isoformat(),
                        "synthetic_outcome_code": 3 if member and (i + visit) % 2 else 1,
                    }
                )
    return alerts


def toy_grid(anchor, index, seed=17):
    """Eight days plus one future hour; future data tests the causal boundary."""
    start = pd.Timestamp(anchor).value - 8 * 24 * 3600_000_000_000
    tick = np.arange(8 * 24 * 12 + 12)
    rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
    grid = np.empty((len(tick), len(COLUMNS)), dtype=float)
    grid[:, 0] = 60 + (tick + index) % 13 + rng.normal(0, 0.1, len(tick))
    grid[:, 1] = 110 + (2 * tick + index) % 17
    grid[:, 2] = 65 + (tick + 2 * index) % 11
    grid[:, 3] = 94 + (tick + index) % 5
    grid[:, 4] = 36 + ((tick + index) % 7) / 10
    for channel in range(5):
        grid[(tick + index + 3 * channel) % (11 + channel) == 0, channel] = np.nan
    grid[:, 5] = grid[:, 1] - grid[:, 2]
    grid[:, 6] = ((tick + index) % 6) * 10
    grid[:, 7] = 5
    grid[:, 8] = 1
    absent = (tick + 2 * index) % 9 == 0
    grid[absent, 6:8] = np.nan
    grid[absent, 8] = 0
    starts = start + tick * STEP_NS
    contributors = {c: starts[np.isfinite(grid[:, i])].tolist() for i, c in enumerate(CHANNELS)}
    return grid, start, contributors


def build_fixture(config, seed=17, patients=12):
    alerts = generate_alerts(patients)
    episodes = group_episodes(alerts)
    cfg = dict(config, _canonical_feature_names=list(COLUMNS))
    rows, records, grids = [], {}, {}
    for i, episode in enumerate(episodes):
        grid, start, contributors = toy_grid(episode["anchor"], i, seed)
        row = episode_statistics(episode["patient"], episode, grid, start, contributors, cfg)
        offsets = (slot_starts(episode["anchor"]) - start) // STEP_NS
        records[row["episode_id"]] = temporal_record(row, grid[offsets, 6])
        grids[row["episode_id"]] = grid
        rows.append(row)
    return alerts, episodes, rows, records, grids
