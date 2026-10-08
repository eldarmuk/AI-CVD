"""Note-free episode grouping and deterministic subject isolation."""

import hashlib
import json

import numpy as np
import pandas as pd

SYNTHETIC_PREFIX = "SYNTHETIC_NOT_A_PATIENT_"


def group_episodes(alerts, burst_minutes=10):
    """Chain adjacent alerts at gaps <= 10 minutes, independently per subject.

    Outcomes are explicit fictional codes. No note corpus or label lexicon is used.
    Normalize UTC timestamps before sorting, including mixed-offset inputs.
    """
    if burst_minutes <= 0:
        raise ValueError("Positive burst duration required")
    clean, seen = [], set()
    for alert in alerts:
        patient = alert["synthetic_subject"]
        if (
            not isinstance(patient, str)
            or not patient.startswith(SYNTHETIC_PREFIX)
            or alert.get("synthetic") is not True
        ):
            raise ValueError("Only explicitly fictional subjects are accepted")
        identifier = alert["synthetic_alert_id"]
        if not isinstance(identifier, str) or not identifier.startswith("SYNTHETIC_ALERT_"):
            raise ValueError("Fictional alert identifier required")
        if (patient, identifier) in seen:
            raise ValueError("Duplicate alert identifier")
        seen.add((patient, identifier))
        at = pd.Timestamp(alert["timestamp"])
        if pd.isna(at) or at.tzinfo is None:
            raise ValueError("Timezone-aware timestamp required")
        code = alert["synthetic_outcome_code"]
        if type(code) is not int or code not in (0, 1, 2, 3):
            raise ValueError("Outcome must be an integer code from 0 to 3")
        clean.append((patient, at.tz_convert("UTC"), identifier, code))
    groups = []
    for alert in sorted(clean):
        if (
            not groups
            or groups[-1][-1][0] != alert[0]
            or alert[1] - groups[-1][-1][1] > pd.Timedelta(minutes=burst_minutes)
        ):
            groups.append([])
        groups[-1].append(alert)
    result = []
    for group in groups:
        patient, at, identifier, _ = group[0]
        anchor = at.isoformat().replace("+00:00", "Z")
        identity = json.dumps([patient, anchor, identifier])
        result.append(
            {
                "synthetic": True,
                "patient": patient,
                "anchor": anchor,
                "episode_id": "synthetic_episode_"
                + hashlib.sha256(identity.encode()).hexdigest()[:20],
                "label": max(x[3] for x in group),
                "constituent_count": len(group),
            }
        )
    return result


def split_patients(episodes, seed=17):
    """Allocate subjects, never individual episodes, to train/validation/test."""
    patients = sorted({r["patient"] for r in episodes})
    if len(patients) < 6:
        raise ValueError("At least six fictional subjects required")
    if any(not p.startswith(SYNTHETIC_PREFIX) for p in patients):
        raise ValueError("Fictional subjects required")
    patients = np.random.default_rng(seed).permutation(patients).tolist()
    held = max(1, len(patients) // 6)
    groups = {
        "train": set(patients[: -2 * held]),
        "validation": set(patients[-2 * held : -held]),
        "test": set(patients[-held:]),
    }
    return {
        r["episode_id"]: next(s for s, ids in groups.items() if r["patient"] in ids)
        for r in episodes
    }
