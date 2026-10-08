"""Retained vectorized causal features; immediate availability only."""

import numpy as np
import pandas as pd

from .features import FEATURE_NAMES, PHYSIOLOGY, STATIC, validate_source_contract


def vector_features(clean, coverage, task, contract):
    validate_source_contract(contract)
    if (
        contract["measurement_availability"] != "timestamp_is_available_at"
        or contract["clinical_history_policy"] != "exclude"
    ):
        raise ValueError("Vectorized path requires immediate availability and excluded history")
    raw = clean.copy()
    parsed = pd.to_datetime(raw["date"], format="mixed")
    if parsed.dt.tz is None:
        parsed = parsed.dt.tz_localize(
            contract["source_timezone"], ambiguous="raise", nonexistent="raise"
        )
    raw["at"] = parsed.dt.tz_convert("UTC")
    start = pd.Timestamp(coverage.get("support_start", coverage.get("enrollment_time")))
    end = pd.Timestamp(coverage["measurement_coverage_end"])
    freq = f"{task.grid_minutes}min"
    raw = raw[(raw["at"] >= start) & (raw["at"] < end)].copy()
    raw["bucket"] = raw["at"].dt.floor(freq)
    first, last = start.ceil(freq), end.floor(freq)
    grid = pd.date_range(first, last, freq=freq, inclusive="left").as_unit("ns")
    frame = pd.DataFrame(index=grid)
    for name in PHYSIOLOGY:
        if name == "steps":
            continue
        values = raw[raw[name].notna()]
        frame[name] = values.groupby("bucket")[name].mean().reindex(grid)
        latest = values.groupby("bucket").at.max().reindex(grid).ffill()
        frame["time_since_last_" + name] = (
            pd.Series(grid + pd.Timedelta(freq), index=grid) - latest
        ).dt.total_seconds() / 60
    steps = raw[raw.steps.notna()].copy()
    frame["steps_source_value"] = steps.groupby("bucket").steps.last().reindex(grid)
    frame["observed_steps_source"] = frame.steps_source_value.notna().astype(float)
    steps["interval"] = np.nan
    steps["reset"] = False
    if contract["steps_semantics"] == "cumulative_counter":
        previous = steps.steps.shift()
        delta = steps.steps - previous
        days = steps["at"].dt.tz_convert(contract["source_timezone"]).dt.date
        steps["reset"] = previous.notna() & (
            (delta < 0) | ((days != days.shift()) & (contract["counter_reset_policy"] == "daily"))
        )
        steps["interval"] = steps["at"].diff().dt.total_seconds() / 60
        steps["steps"] = delta.where(
            previous.notna() & ~steps.reset & delta.between(*task.validity["steps"])
        )
        steps.loc[steps.steps.isna(), "interval"] = np.nan
    frame["steps_counter_reset"] = (
        steps.groupby("bucket")
        .reset.max()
        .reindex(grid)
        .astype("boolean")
        .fillna(False)
        .astype(float)
    )
    frame["steps_delta_interval_minutes"] = steps.groupby("bucket").interval.max().reindex(grid)
    increments = steps[steps.steps.notna()]
    frame["steps"] = increments.groupby("bucket").steps.sum(min_count=1).reindex(grid)
    latest = increments.groupby("bucket").at.max().reindex(grid).ffill()
    frame["time_since_last_steps"] = (
        pd.Series(grid + pd.Timedelta(freq), index=grid) - latest
    ).dt.total_seconds() / 60
    for name in PHYSIOLOGY:
        frame["observed_" + name] = frame[name].notna().astype(float)
    frame["shock_index"] = frame.heartrate / frame.sbp
    frame["hr_bucket_sd_4h"] = frame.heartrate.rolling(
        task.hr_sd_minutes // task.grid_minutes, min_periods=2
    ).std(ddof=1)
    n = task.bp_trend_minutes // task.grid_minutes
    x = pd.Series(np.arange(len(grid)) * task.grid_minutes / 60, index=grid).where(
        frame.sbp.notna()
    )
    y = frame.sbp
    count = y.rolling(n, min_periods=2).count()
    sx, sy = x.rolling(n, min_periods=2).sum(), y.rolling(n, min_periods=2).sum()
    denom = (x * x).rolling(n, min_periods=2).sum() - sx * sx / count
    frame["bp_trend_mmhg_per_hour"] = (
        (x * y).rolling(n, min_periods=2).sum() - sx * sy / count
    ) / denom.where(denom > 1e-10)
    n = task.steps_sum_minutes // task.grid_minutes
    frame["steps_sum_6h"] = frame.steps.rolling(n, min_periods=1).sum()
    frame["steps_observed_buckets_6h"] = (
        frame.steps.notna().astype(float).rolling(n, min_periods=1).sum()
    )
    hour = grid.hour + grid.minute / 60
    frame["hour_sin"], frame["hour_cos"] = (
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
    )
    frame["is_night"] = (hour < 6).astype(float)
    for name in STATIC:
        frame[name], frame["known_" + name] = np.nan, 0.0
    return frame.loc[:, list(FEATURE_NAMES)].astype(float)
