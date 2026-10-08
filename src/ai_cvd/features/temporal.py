"""Causal 288-slot adapter and partition-local population statistics."""

import numpy as np
import pandas as pd

from ai_cvd.features.episode import CHANNELS, STEP_NS, PartitionPreprocessor

CHANNEL_ORDER = (*CHANNELS, "steps")


def slot_starts(anchor):
    at = pd.Timestamp(anchor)
    if at.tzinfo is None:
        raise ValueError("Timezone required")
    # Exactly 287 closed admitted slots and one explicit leading padding slot.
    first = ((at.value - 24 * 3600_000_000_000 + STEP_NS - 1) // STEP_NS) * STEP_NS
    starts = first + np.arange(287, dtype=np.int64) * STEP_NS
    if starts[-1] + STEP_NS >= at.value:
        raise ValueError("Causal grid error")
    return starts


def temporal_record(row, steps):
    starts = slot_starts(row["anchor"])
    values = np.full((288, 7), np.nan)
    lookup = {int(t): i + 1 for i, t in enumerate(starts)}
    seen = set()
    for patient, t, channel, value in row["recent_buckets"]:
        key = (int(t), channel)
        if (
            patient != row["patient"]
            or t not in lookup
            or channel not in CHANNELS
            or key in seen
            or not np.isfinite(value)
        ):
            raise ValueError("Invalid/duplicate B1 bucket")
        seen.add(key)
        values[lookup[t], CHANNELS.index(channel)] = value
    if row["admitted_recent_buckets"] != 287:
        raise ValueError("B1 recent grid coverage differs")
    if len(steps) != 287:
        raise ValueError("Incomplete Steps history")
    values[1:, 6] = steps
    return {
        "values": values,
        "starts": np.r_[starts[0] - STEP_NS, starts],
        "admitted": np.r_[False, np.ones(287, dtype=bool)],
    }


def elapsed(mask, admitted):
    """Hours since preceding observation, or admitted history start if unknown."""
    delta = np.zeros(mask.shape, dtype=np.float32)
    for t in range(1, len(mask)):
        if admitted[t] and admitted[t - 1]:
            delta[t] = 1 / 12 + (1 - mask[t - 1]) * delta[t - 1]
    return delta


class TemporalPreprocessor:
    def __init__(self, config):
        self.config = config

    def fit(self, rows, records, *, fitting_patients, forbidden_patients=()):
        self.context = PartitionPreprocessor(self.config).fit(
            rows, fitting_patients=fitting_patients, forbidden_patients=forbidden_patients
        )
        unique = {}
        for row in rows:
            rec = records[row["episode_id"]]
            for t in np.flatnonzero(rec["admitted"]):
                for c, v in enumerate(rec["values"][t]):
                    if np.isfinite(v):
                        key = (row["patient"], int(rec["starts"][t]), c)
                        if key in unique and unique[key] != v:
                            raise ValueError("Conflicting temporal bucket")
                        unique[key] = float(v)
        self.mean, self.scale = np.zeros(7), np.ones(7)
        for c in range(7):
            v = np.array([v for (_, _, k), v in unique.items() if k == c])
            if len(v):
                self.mean[c] = v.mean()
                sd = v.std(ddof=0)
                self.scale[c] = sd if sd > 0 else 1
        return self

    def transform(self, rows, records, family):
        if family not in ("grud", "relative_grud"):
            raise ValueError("Unknown family")
        vals, masks, deltas, admitted, targets = [], [], [], [], []
        for row in rows:
            rec = records[row["episode_id"]]
            raw = rec["values"]
            mask = np.isfinite(raw) & rec["admitted"][:, None]
            x = np.where(mask, (raw - self.mean) / self.scale, 0.0)
            target = np.zeros(7)
            if family == "relative_grud":
                relative, rm, rt = np.zeros((288, 6)), np.zeros((288, 6), bool), np.zeros(6)
                for c, channel in enumerate(CHANNELS):
                    base = row["baseline"][channel]
                    if base["usable"]:
                        scale = (
                            1.4826 * base["mad"]
                            if base["mad"] > 0
                            else self.context.fallback[channel]
                        )
                        rm[:, c] = mask[:, c]
                        relative[:, c] = np.where(
                            rm[:, c], (raw[:, c] - base["median"]) / scale, 0.0
                        )
                        # Study B relative-input convention: transform the
                        # partition absolute mean by the same personal baseline.
                        rt[c] = (self.mean[c] - base["median"]) / scale
                x, mask, target = np.c_[x, relative], np.c_[mask, rm], np.r_[target, rt]
            vals.append(x)
            masks.append(mask)
            targets.append(target)
            admitted.append(rec["admitted"])
            deltas.append(elapsed(mask, rec["admitted"]))
        result = dict(
            values=np.asarray(vals, np.float32),
            masks=np.asarray(masks, np.float32),
            deltas=np.asarray(deltas, np.float32),
            admitted=np.asarray(admitted, bool),
            target=np.asarray(targets, np.float32),
            context=self.context.transform(
                rows, self.config["models"]["neural_context_subset"]
            ).astype(np.float32),
        )
        if any(not np.isfinite(v).all() for v in result.values()):
            raise ValueError("Nonfinite temporal preprocessing")
        return result

    def state(self):
        return {
            "context": self.context.state(),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "relative_target": "personal_transform_of_partition_absolute_mean",
            "elapsed_unit": "hours",
        }
