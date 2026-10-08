"""Outcome-independent candidate windows, explicit censoring, and ID-safe joining."""

import hashlib
import json
from collections import Counter
from datetime import timedelta

from .episodes import as_time, event_time, iso
from .features import FEATURE_NAMES, validate_coverage


def sample_id(senior_id, prediction_time):
    # Task-independent identity; the task fingerprint disambiguates labels/schemas.
    payload = json.dumps([str(senior_id), iso(prediction_time)], separators=(",", ":"))
    return "sample_" + hashlib.sha256(payload.encode()).hexdigest()


def patient_split(senior_id, task):
    digest = hashlib.sha256(f"{task.split_seed}:{senior_id}".encode()).digest()
    fraction = int.from_bytes(digest[:8], "big") / 2**64
    train, validation, _ = task.split_ratios
    return (
        "train" if fraction < train else "validation" if fraction < train + validation else "test"
    )


def assert_patient_isolation(manifests):
    assignments = {}
    for row in manifests:
        sid, split = str(row["senior_id"]), row["split"]
        if split not in {"train", "validation", "test"}:
            raise ValueError("Unknown split")
        if sid in assignments and assignments[sid] != split:
            raise ValueError("Patient overlap across splits")
        assignments[sid] = split


def align_by_sample_id(expected_ids, records, *, task_identifier, endpoint, source_manifest_sha256):
    """Strict one-to-one keyed join, not a length/order assertion."""
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("Duplicate expected sample IDs")
    lookup = {}
    for row in records:
        key = row["sample_id"]
        if key in lookup:
            raise ValueError("Duplicate record sample ID")
        if (
            row["task_identifier"] != task_identifier
            or row["endpoint"] != endpoint
            or row.get("source_manifest_sha256") != source_manifest_sha256
            or not source_manifest_sha256
        ):
            raise ValueError("Task/endpoint/source manifest mismatch during sample join")
        lookup[key] = row
    if set(lookup) != set(expected_ids):
        raise ValueError("Missing or unexpected sample IDs")
    return [lookup[key] for key in expected_ids]


def sequence_for_sample(features, sample, task):
    if sample["task_identifier"] != task.identifier:
        raise ValueError("Task fingerprint mismatch")
    t = as_time(sample["prediction_time"])
    first = t - timedelta(minutes=task.lookback_minutes)
    if sample["sample_id"] != sample_id(sample["senior_id"], t):
        raise ValueError("Immutable sample identity mismatch")
    if as_time(sample["input_start"]) != first or as_time(sample["input_end_exclusive"]) != t:
        raise ValueError("Input interval mismatch")
    rows = [r for r in features if first <= as_time(r["timestamp"]) < t]
    if len(rows) != task.sequence_steps:
        raise ValueError("Incomplete sequence")
    expected = [
        first + timedelta(minutes=i * task.grid_minutes) for i in range(task.sequence_steps)
    ]
    if [as_time(r["timestamp"]) for r in rows] != expected:
        raise ValueError("Sequence grid is not contiguous and ordered")
    if any(
        str(r["senior_id"]) != str(sample["senior_id"]) or as_time(r["available_at"]) > t
        for r in rows
    ):
        raise ValueError("Cross-patient or future information in input")
    if any(
        as_time(r["available_at"]) != as_time(r["timestamp"]) + timedelta(minutes=task.grid_minutes)
        for r in rows
    ):
        raise ValueError("Bucket closure mismatch")
    # Fixed allowlist; labels/IDs/timestamps can never become neural features.
    return [[r[name] for name in FEATURE_NAMES] for r in rows]


def generate_patient_manifest(features, episodes, coverage, task, endpoint=None):
    times = validate_coverage(coverage)
    sid = str(coverage["senior_id"])
    split = patient_split(sid, task)  # Assigned before any candidate construction.
    endpoint = endpoint or task.endpoint
    if endpoint not in {task.endpoint, task.secondary_endpoint}:
        raise ValueError("Endpoint must be explicitly primary or configured secondary")
    severities = task.target_severities if endpoint == task.endpoint else task.secondary_severities
    events = sorted(
        [
            (event_time(e, severities), e["episode_id"])
            for e in episodes
            if str(e["senior_id"]) == sid and event_time(e, severities) is not None
        ]
    )
    run_in_end = times["enrollment_time"] + timedelta(minutes=task.run_in_minutes)
    run_in = [
        r
        for r in features
        if as_time(r["timestamp"]) >= times["enrollment_time"]
        and as_time(r["available_at"]) <= run_in_end
    ]
    hr = sum(r["observed_heartrate"] for r in run_in)
    bp = sum(r["observed_pulse_pressure"] for r in run_in)
    density_ok = hr >= task.min_valid_hr_per_run_in and bp >= task.min_valid_bp_per_run_in
    counts, stream, event_flags = (
        Counter(),
        [],
        {eid: {"eligible": 0, "reasons": set()} for _, eid in events},
    )
    width = timedelta(minutes=task.grid_minutes)
    lookback = timedelta(minutes=task.lookback_minutes)
    horizon = timedelta(minutes=task.horizon_minutes)
    for index, feature in enumerate(features):
        if index % task.prediction_stride_steps:
            continue
        t = as_time(feature["available_at"])
        counts["candidate_prediction_times"] += 1
        # Eligibility uses only run-in observations and an external coverage contract.
        reasons = []
        if index + 1 < task.sequence_steps or t - lookback < times["enrollment_time"]:
            reasons.append("insufficient_history")
        if t < run_in_end:
            reasons.append("run_in_incomplete")
        elif not density_ok:
            reasons.append("insufficient_run_in_density")
        if times["outcome_coverage_start"] > t:
            reasons.append("outcome_coverage_not_started")
        if min(times["measurement_coverage_end"], times["outcome_coverage_end"]) < t + horizon:
            reasons.append("insufficient_follow_up")
        matched = [(at, eid) for at, eid in events if t < at <= t + horizon]
        for _, eid in matched:
            event_flags[eid]["reasons"].update(reasons)
            event_flags[eid]["eligible"] += not reasons
        if reasons:
            counts["excluded_" + reasons[0]] += 1  # Disjoint sequential flow counts.
            continue
        selected = features[index + 1 - task.sequence_steps : index + 1]
        expected = [t - lookback + j * width for j in range(task.sequence_steps)]
        if [as_time(r["timestamp"]) for r in selected] != expected:
            raise ValueError("Noncontiguous feature timeline")
        if any(str(r["senior_id"]) != sid or as_time(r["available_at"]) > t for r in selected):
            raise ValueError("Feature causality/patient violation")
        row = {
            "sample_id": sample_id(sid, t),
            "senior_id": sid,
            "prediction_time": iso(t),
            "input_start": iso(t - lookback),
            "input_end_exclusive": iso(t),
            "target": int(bool(matched)),
            "episode_id": matched[0][1] if matched else None,
            "episode_ids": [eid for _, eid in matched],
            "lead_minutes": (matched[0][0] - t).total_seconds() / 60 if matched else None,
            "lead_time_definition": "minutes_to_alarm_initiation",
            "split": split,
            "task_identifier": task.identifier,
            "endpoint": endpoint,
            "sampling": "complete_eligible_stream",
            "feature_schema": task.feature_schema_version,
        }
        stream.append(row)
    counts["eligible_prediction_times"] = len(stream)
    counts["eligible_patients"] = int(bool(stream))
    counts["patients_initially_available"] = 1
    counts["events_total"] = len(events)
    counts["events_with_eligible_prediction"] = sum(v["eligible"] > 0 for v in event_flags.values())
    counts["events_without_eligible_prediction"] = sum(
        v["eligible"] == 0 for v in event_flags.values()
    )
    for reason in (
        "insufficient_history",
        "insufficient_follow_up",
        "run_in_incomplete",
        "insufficient_run_in_density",
        "outcome_coverage_not_started",
    ):
        counts["events_lost_" + reason] = sum(
            v["eligible"] == 0 and reason in v["reasons"] for v in event_flags.values()
        )
        counts["patients_excluded_" + reason] = int(not stream and counts["excluded_" + reason] > 0)
    counts["events_outside_candidate_horizons"] = sum(
        v["eligible"] == 0 and not v["reasons"] for v in event_flags.values()
    )
    return stream, dict(counts)


def training_manifest(stream, episodes, task, policy=None, coverage=None):
    """Explicit retrospective training selection; never alters stream eligibility."""
    policy = policy or task.training_policy
    if coverage is None:
        raise ValueError("Retrospective training requires explicit outcome coverage")
    if policy not in {"event_free_windows", "never_event_patients"}:
        raise ValueError("Unknown training policy")
    by_patient = {}
    for e in episodes:
        for member in e["constituents"]:
            if member["severity"] in task.training_excluded_severities:
                by_patient.setdefault(str(e["senior_id"]), []).append(as_time(member["timestamp"]))
    result = []
    for row in stream:
        if row["split"] != "train":
            continue
        t = as_time(row["prediction_time"])
        start, end = (
            t - timedelta(minutes=task.lookback_minutes),
            t + timedelta(minutes=task.horizon_minutes),
        )
        if str(coverage["senior_id"]) != str(row["senior_id"]):
            raise ValueError("Cross-patient training coverage")
        if (
            as_time(coverage["outcome_coverage_start"]) > start
            or as_time(coverage["outcome_coverage_end"]) < end
        ):
            continue
        patient_events = by_patient.get(str(row["senior_id"]), [])
        excluded = (
            bool(patient_events)
            if policy == "never_event_patients"
            else any(start <= at <= end for at in patient_events)
        )
        if not excluded:
            result.append(
                dict(row, sampling="retrospective_unsupervised_training", training_policy=policy)
            )
    return result


def development_manifest(stream, per_class=1000, seed=42):
    """Optional balanced train/validation subset; held-out test is never sampled here."""
    if per_class <= 0:
        raise ValueError("per_class must be positive")
    groups = {}
    for row in stream:
        if row["split"] == "test":
            continue
        groups.setdefault((row["split"], row["target"]), []).append(row)
    selected = []
    for split in ("train", "validation"):
        size = min(per_class, len(groups.get((split, 0), [])), len(groups.get((split, 1), [])))
        for label in (0, 1):
            pool = sorted(
                groups.get((split, label), []),
                key=lambda r: hashlib.sha256(f"{seed}:{r['sample_id']}".encode()).digest(),
            )
            selected.extend(
                dict(r, sampling="balanced_development_not_natural_prevalence") for r in pool[:size]
            )
    return selected
