"""Study B closure-time adapter over in-memory temporal records."""

import numpy as np
import pandas as pd

from ai_cvd.features.episode import STEP_NS
from ai_cvd.features.temporal import TemporalPreprocessor, slot_starts


class MTANPreprocessor(TemporalPreprocessor):
    def fit(self, rows, records, *, fitting_patients, forbidden_patients=()):
        for row in rows:
            self.closure_times(row, records[row["episode_id"]])
        return super().fit(
            rows, records, fitting_patients=fitting_patients, forbidden_patients=forbidden_patients
        )

    @staticmethod
    def closure_times(row, rec):
        anchor = pd.Timestamp(row["anchor"]).value
        expected = slot_starts(row["anchor"])
        if (
            rec["values"].shape != (288, 7)
            or rec["starts"].shape != (288,)
            or rec["admitted"].shape != (288,)
        ):
            raise ValueError("Frozen temporal dimensions differ")
        if (
            not np.array_equal(rec["starts"][1:], expected)
            or rec["admitted"][0]
            or not rec["admitted"][1:].all()
        ):
            raise ValueError("Temporal grid/padding differs from the frozen interface")
        ends = rec["starts"] + STEP_NS
        if np.any(ends[rec["admitted"]] >= anchor):
            raise ValueError("At/after-alarm key is forbidden")
        return np.where(rec["admitted"], (ends - anchor) / (24 * 3600_000_000_000), 0.0)

    def transform(self, rows, records, family):
        if family != "mtan":
            raise ValueError("B3 supports only the frozen mTAN family")
        # Reuse exactly B2 absolute channel statistics and common context transforms.
        base = super().transform(rows, records, "grud")
        keys, references = [], []
        for row in rows:
            rec = records[row["episode_id"]]
            times = self.closure_times(row, rec)
            keys.append(times)
            references.append(np.linspace(-1.0, times[-1], 48))
        return {k: base[k] for k in ("values", "masks", "admitted", "context")} | {
            "key_times": np.asarray(keys, np.float32),
            "reference_times": np.asarray(references, np.float32),
        }

    def state(self):
        return super().state() | {
            "B3_time_convention": "bucket_closure_minus_anchor_div_24h",
            "B3_attention_scope": "all_observed_admitted_pre_alarm_keys_per_reference",
            "B3_reference_count": 48,
            "B3_channels": "seven_absolute_only",
        }
