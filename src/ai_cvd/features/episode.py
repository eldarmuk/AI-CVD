"""Deterministic Study B episode features and partition-local preprocessing."""

from __future__ import annotations

import math
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

CHANNELS = ("heartrate", "sbp", "dbp", "saturation", "temperature", "pulse_pressure")
STEP_NS = 300_000_000_000


def configuration(path):
    with Path(path).open("rb") as fh:
        return tomllib.load(fh)


def finite(values):
    a = np.asarray(values, dtype=float)
    return a[np.isfinite(a)]


def quantile(values, q):
    a = finite(values)
    return float(np.quantile(a, q, method="linear")) if len(a) else math.nan


def iqr(values):
    a = finite(values)
    return float(np.diff(np.quantile(a, [0.25, 0.75], method="linear"))[0]) if len(a) else math.nan


def longest_false_run(mask):
    longest = current = 0
    for present in mask:
        current = 0 if present else current + 1
        longest = max(longest, current)
    return longest * 5


def _utc_ns(value):
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise ValueError("Alarm and bucket timestamps must carry a timezone")
    return stamp.tz_convert("UTC").value


def episode_statistics(
    patient, episode, grid, grid_start_ns, contributor_times, config, history_hours=None
):
    """Pure feature extraction from canonical, closed five-minute buckets.

    `contributor_times` maps channel to canonical valid contributor UTC nanoseconds.
    It is used only for recency. A caller must verify coverage of the entire
    baseline and recent interval before calling this function.
    """
    names = config["features"]["channel_order"]
    if list(names) != list(CHANNELS):
        raise ValueError("Unexpected frozen channel order")
    anchor = _utc_ns(episode["anchor"])
    h = int(history_hours or config["history"]["primary_hours"])
    if h not in (
        int(config["history"]["primary_hours"]),
        int(config["history"]["sensitivity_hours"]),
    ):
        raise ValueError("Only frozen 24h/8h histories are allowed")
    starts = grid_start_ns + np.arange(len(grid), dtype=np.int64) * STEP_NS
    recent = (starts >= anchor - h * 3600_000_000_000) & (starts + STEP_NS < anchor)
    base = (starts >= anchor - 8 * 86400_000_000_000) & (
        starts + STEP_NS <= anchor - 24 * 3600_000_000_000
    )
    if not np.any(recent):
        raise ValueError("No admitted recent buckets")
    if np.any(starts[recent] + STEP_NS >= anchor):
        raise ValueError("Alarm-time bucket admitted")
    idx = {name: i for i, name in enumerate(config["_canonical_feature_names"])}
    result, baseline, union = {}, {}, []
    recent_starts = starts[recent]
    base_starts = starts[base]
    for channel in CHANNELS:
        rv = np.asarray(grid[recent, idx[channel]], dtype=float)
        bv = np.asarray(grid[base, idx[channel]], dtype=float)
        observed = np.isfinite(rv)
        bobs = np.isfinite(bv)
        good = finite(rv)
        bvalid = finite(bv)
        local_days = len(
            set(pd.to_datetime(base_starts[bobs], utc=True).tz_convert("Europe/Warsaw").date)
        )
        usable = len(bvalid) >= 12 and local_days >= 3
        bmedian = quantile(bvalid, 0.5)
        bmad = quantile(np.abs(bvalid - bmedian), 0.5) if len(bvalid) else math.nan
        biqr = iqr(bvalid)
        baseline[channel] = {
            "median": bmedian,
            "mad": bmad,
            "iqr": biqr,
            "count": len(bvalid),
            "local_days": local_days,
            "usable": bool(usable),
        }
        for label, window in (("1h", 1), ("6h", 6), ("H", h)):
            values = rv[recent_starts >= anchor - window * 3600_000_000_000]
            result[f"{channel}__median_{label}"] = quantile(values, 0.5)
        result[f"{channel}__last_H"] = (
            float(rv[np.flatnonzero(observed)[-1]]) if observed.any() else math.nan
        )
        median_6 = result[f"{channel}__median_6h"]
        result[f"{channel}__delta_median_6h"] = (
            median_6 - bmedian if usable and np.isfinite(median_6) else math.nan
        )
        result[f"{channel}__robust_z_6h"] = math.nan  # deferred to fitting partition
        result[f"{channel}__delta_iqr_H"] = (
            iqr(good) - biqr if usable and len(good) >= 4 else math.nan
        )
        fraction = float(observed.mean())
        result[f"{channel}__delta_observation_rate_H"] = (
            fraction - len(bvalid) / 2016 if usable else math.nan
        )
        result[f"{channel}__valid_fraction_H"] = fraction
        result[f"{channel}__longest_gap_H"] = longest_false_run(observed)
        last_end = int(recent_starts[-1] + STEP_NS)
        times = np.asarray(contributor_times.get(channel, []), dtype=np.int64)
        prior = times[(times >= anchor - h * 3600_000_000_000) & (times < last_end)]
        result[f"{channel}__recency_H"] = (
            min(h * 60.0, (anchor - int(prior.max())) / 60_000_000_000) if len(prior) else h * 60.0
        )
        result[f"{channel}__baseline_usable"] = float(usable)
        result[f"{channel}__baseline_valid_count"] = float(min(len(bvalid), 2016))
        result[f"{channel}__baseline_scale_valid"] = float(
            usable and np.isfinite(bmad) and bmad > 0
        )
        for at, value in zip(recent_starts[observed], good):
            union.append((str(patient), int(at), channel, float(value)))
    steps = np.asarray(grid[recent, idx["steps"]], dtype=float)
    increments = finite(steps)
    result["steps__sum_H"] = float(increments.sum()) if len(increments) else math.nan
    result["steps__increment_fraction_H"] = float(np.isfinite(steps).mean())
    intervals = finite(grid[recent, idx["steps_delta_interval_minutes"]])
    result["steps__max_delta_interval_H"] = float(intervals.max()) if len(intervals) else math.nan
    result["steps__source_fraction_H"] = float(
        np.asarray(grid[recent, idx["observed_steps_source"]], dtype=float).sum() / recent.sum()
    )
    local = pd.Timestamp(anchor, tz="UTC").tz_convert("Europe/Warsaw")
    hour = local.hour + local.minute / 60 + local.second / 3600 + local.microsecond / 3_600_000_000
    result["clock__sin"] = math.sin(2 * math.pi * hour / 24)
    result["clock__cos"] = math.cos(2 * math.pi * hour / 24)
    expected = [f["name"] for f in config["features"]["fields"]]
    if set(result) != set(expected):
        raise ValueError(f"Feature schema mismatch: {set(expected) ^ set(result)}")
    return {
        "episode_id": episode["episode_id"],
        "patient": str(patient),
        "anchor": episode["anchor"],
        "label": int(episode["label"]),
        "features": {n: result[n] for n in expected},
        "baseline": baseline,
        "recent_buckets": union,
        "admitted_recent_buckets": int(recent.sum()),
        "admitted_baseline_buckets": int(base.sum()),
    }


def fallback_scales(rows, channels=CHANNELS):
    unique = {}
    for row in rows:
        for patient, timestamp, channel, value in row["recent_buckets"]:
            if patient != row["patient"] or channel not in channels or not np.isfinite(value):
                raise ValueError("Invalid canonical bucket provenance")
            key = (patient, int(timestamp), channel)
            if key in unique and unique[key] != value:
                raise ValueError("Conflicting canonical bucket key")
            unique[key] = value
    result = {}
    for channel in channels:
        values = [v for (_, _, c), v in unique.items() if c == channel]
        width = iqr(values) / 1.349
        result[channel] = float(width) if np.isfinite(width) and width > 0 else 1.0
    return result


def materialize(row, fallback, config):
    result = dict(row["features"])
    for channel in CHANNELS:
        base = row["baseline"][channel]
        delta = result[f"{channel}__delta_median_6h"]
        mad = base["mad"]
        scale = 1.4826 * mad if np.isfinite(mad) and mad > 0 else fallback[channel]
        result[f"{channel}__robust_z_6h"] = (
            delta / scale if base["usable"] and np.isfinite(delta) else math.nan
        )
    for field in config["features"]["fields"]:
        name = field["name"]
        value = result[name]
        if field["transform"] == "log1p" and np.isfinite(value):
            if value < 0:
                raise ValueError("Negative log1p input: " + name)
            result[name] = math.log1p(value)
    return result


class PartitionPreprocessor:
    """Fit accepts only the supplied fitting partition; transform is read-only."""

    def __init__(self, config):
        self.config = config
        self.fallback = None
        self.parameters = None

    def fit(self, fitting_rows, *, fitting_patients, forbidden_patients=()):
        rows = list(fitting_rows)
        if not rows or len({r["episode_id"] for r in rows}) != len(rows):
            raise ValueError("Empty or duplicate fitting episodes")
        fitting_patients = set(fitting_patients)
        forbidden_patients = set(forbidden_patients)
        if not fitting_patients or fitting_patients & forbidden_patients:
            raise ValueError("Invalid fitting/held-out patient partition")
        if {r["patient"] for r in rows} != fitting_patients or any(
            r["patient"] in forbidden_patients for r in rows
        ):
            raise ValueError("Fitting rows do not exactly match authorized training patients")
        self.fitting_ids = frozenset(r["episode_id"] for r in rows)
        self.fitting_patients = frozenset(fitting_patients)
        self.fallback = fallback_scales(rows)
        records = [materialize(r, self.fallback, self.config) for r in rows]
        self.parameters = {}
        for field in self.config["features"]["fields"]:
            name = field["name"]
            values = finite([r[name] for r in records])
            identity = name.startswith("clock__") or name.endswith(
                ("__baseline_usable", "__baseline_scale_valid")
            )
            median = float(np.median(values)) if len(values) else 0.0
            spread = iqr(values) / 1.349
            self.parameters[name] = {
                "impute": median,
                "center": 0.0 if identity else median,
                "scale": 1.0
                if identity or not np.isfinite(spread) or spread <= 0
                else float(spread),
            }
        return self

    def transform(self, rows, subset):
        if self.parameters is None:
            raise ValueError("Preprocessor is not fitted")
        spec = self.config["feature_subsets"][subset]
        names = spec["raw_names"]
        flags = spec["missing_indicator_names"]
        output = np.empty((len(rows), len(names) + len(flags)), dtype=float)
        for i, row in enumerate(rows):
            record = materialize(row, self.fallback, self.config)
            for j, name in enumerate(names):
                value = record[name]
                p = self.parameters[name]
                output[i, j] = ((value if np.isfinite(value) else p["impute"]) - p["center"]) / p[
                    "scale"
                ]
            for j, flag in enumerate(flags, len(names)):
                output[i, j] = float(not np.isfinite(record[flag.removesuffix("__missing")]))
        if not np.isfinite(output).all():
            raise ValueError("Nonfinite encoded matrix")
        return output

    def state(self):
        return {
            "fit_episode_ids": sorted(self.fitting_ids),
            "fit_patients": sorted(self.fitting_patients),
            "fallback": self.fallback,
            "parameters": self.parameters,
        }
