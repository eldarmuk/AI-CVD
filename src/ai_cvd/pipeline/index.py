"""Retained vectorized continuous-window indexing and cohort flow accounting."""

from collections import Counter

import numpy as np
import pandas as pd

from .episodes import event_time


def window_index(frame, episodes, coverage, task):
    """Vectorized equivalent of the canonical label-independent candidate grid."""
    n = len(frame)
    times = frame.index.asi8 + task.grid_minutes * 60 * 10**9
    minute = 60 * 10**9
    origin = pd.Timestamp(coverage.get("support_start", coverage.get("enrollment_time"))).value
    rin = origin + task.run_in_minutes * minute
    run_in = (frame.index.asi8 >= origin) & (times <= rin)
    density = (
        frame.loc[run_in, "observed_heartrate"].sum() >= task.min_valid_hr_per_run_in
        and frame.loc[run_in, "observed_pulse_pressure"].sum() >= task.min_valid_bp_per_run_in
    )
    reason_names = [
        "insufficient_history",
        "run_in_incomplete",
        "insufficient_run_in_density",
        "outcome_coverage_not_started",
        "insufficient_follow_up",
    ]
    flags = [
        (
            (np.arange(n) + 1 < task.sequence_steps)
            | (times - task.lookback_minutes * minute < origin)
        ),
        times < rin,
        (times >= rin) & ~np.full(n, density),
        times < pd.Timestamp(coverage["outcome_coverage_start"]).value,
        times + task.horizon_minutes * minute
        > min(
            pd.Timestamp(coverage["measurement_coverage_end"]).value,
            pd.Timestamp(coverage["outcome_coverage_end"]).value,
        ),
    ]
    reason = np.full(n, -1, dtype=np.int8)
    for i, flag in enumerate(flags):
        reason[(reason < 0) & flag] = i
    candidates = np.arange(n) % task.prediction_stride_steps == 0
    eligible = (reason < 0) & candidates
    events = sorted(
        [
            (pd.Timestamp(event_time(e, task.target_severities)).value, e["episode_id"])
            for e in episodes
            if event_time(e, task.target_severities) is not None
        ]
    )
    et = np.array([t for t, _ in events], dtype=np.int64)
    first = np.searchsorted(et, times, side="right")
    stop = np.searchsorted(et, times + task.horizon_minutes * minute, side="right")
    target = first < stop
    lead = np.full(n, np.nan)
    lead[target] = (et[first[target]] - times[target]) / minute
    actual = np.array(
        sorted(
            pd.Timestamp(m["timestamp"]).value
            for e in episodes
            for m in e["constituents"]
            if m["severity"] in task.training_excluded_severities
        ),
        dtype=np.int64,
    )
    train_ok = np.searchsorted(
        actual, times + task.horizon_minutes * minute, side="right"
    ) == np.searchsorted(actual, times - task.lookback_minutes * minute, side="left")
    train_ok &= (
        times - task.lookback_minutes * minute
        >= pd.Timestamp(coverage["outcome_coverage_start"]).value
    )
    if task.training_policy == "never_event_patients":
        train_ok &= len(actual) == 0
    flow = Counter(
        candidate_prediction_times=int(candidates.sum()),
        eligible_prediction_times=int(eligible.sum()),
        eligible_patients=int(eligible.any()),
        patients_initially_available=1,
        events_total=len(events),
    )
    for i, name in enumerate(reason_names):
        flow["excluded_" + name] = int(((reason == i) & candidates).sum())
        flow["patients_excluded_" + name] = int(not eligible.any() and flow["excluded_" + name] > 0)
    for at, _ in events:
        opportunities = (times >= at - task.horizon_minutes * minute) & (times < at) & candidates
        found = bool((eligible & opportunities).any())
        flow[
            "events_with_eligible_prediction" if found else "events_without_eligible_prediction"
        ] += 1
        if not found:
            flow["events_outside_candidate_horizons"] += int(not opportunities.any())
            for flag, name in zip(flags, reason_names):
                flow["events_lost_" + name] += int((opportunities & flag).any())
    ends = np.flatnonzero(eligible).astype(np.int32)
    return (
        {
            "end_rows": ends,
            "target": target[eligible].astype(np.uint8),
            "lead_minutes": lead[eligible].astype(np.float32),
            "training": train_ok[eligible].astype(np.uint8),
            "first_event": first[eligible].astype(np.int32),
            "event_stop": stop[eligible].astype(np.int32),
        },
        dict(flow),
        events,
    )
