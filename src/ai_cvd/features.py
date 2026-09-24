"""Causal bucket features, shared by the SQLite and DuckDB source adapters."""
from collections import defaultdict, deque
from datetime import timedelta
import math
import statistics
from zoneinfo import ZoneInfo

from .episodes import as_time, iso

PHYSIOLOGY = ("temperature", "heartrate", "sbp", "dbp", "saturation", "steps", "pulse_pressure")
STATIC = ("age", "gender", "cardiovascular", "metabolic_endocrine", "neurological", "psychiatric_cognitive", "musculoskeletal", "respiratory", "gastro_renal_urologic", "oncological", "sensory", "other_functional_risk", "other")
FEATURE_NAMES = (PHYSIOLOGY + ("steps_source_value", "observed_steps_source", "steps_counter_reset", "steps_delta_interval_minutes", "shock_index", "hr_bucket_sd_4h", "bp_trend_mmhg_per_hour", "steps_sum_6h", "steps_observed_buckets_6h", "hour_sin", "hour_cos", "is_night") +
                 tuple("observed_" + name for name in PHYSIOLOGY) +
                 tuple("time_since_last_" + name for name in PHYSIOLOGY) + STATIC +
                 tuple("known_" + name for name in STATIC))
TYPE_MAP = {"Temperature": "temperature", "Heartrate": "heartrate", "Saturation": "saturation", "Steps": "steps"}


def finite(value):
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def valid_value(name, value, task):
    if not finite(value):
        return None
    low, high = task.validity[name]
    return float(value) if low <= float(value) <= high else None


def pulse_pressure(sbp, dbp, task):
    if sbp is None or dbp is None or dbp >= sbp:
        return None
    difference = sbp - dbp
    return difference if difference >= task.pulse_pressure_min else None


def clean_measurement(row, task):
    """Invalid raw values become NULL, never plausible clipped extremes."""
    kind = row["type"]
    if kind == "BloodPressure":
        sbp = valid_value("sbp", row.get("sbp"), task)
        dbp = valid_value("dbp", row.get("dbp"), task)
        if sbp is not None and dbp is not None and dbp >= sbp:
            sbp = dbp = None
        return {"sbp": sbp, "dbp": dbp, "pulse_pressure": pulse_pressure(sbp, dbp, task)}
    if kind not in TYPE_MAP:
        raise ValueError(f"Unsupported measurement type: {kind}")
    name = TYPE_MAP[kind]
    return {name: valid_value(name, row.get("value"), task)}


def grid_floor(t, task):
    t = as_time(t)
    return t - timedelta(seconds=t.timestamp() % (task.grid_minutes * 60))


def validate_source_contract(contract):
    required = ("steps_semantics", "steps_evidence", "measurement_availability", "availability_evidence", "coverage_evidence", "clinical_history_policy", "alert_time_semantics", "alert_time_evidence")
    if any(not contract.get(k) for k in required):
        raise ValueError("Explicit source semantics, availability, coverage evidence, and history policy are required")
    for key in ("steps_evidence", "availability_evidence", "coverage_evidence", "alert_time_evidence"):
        if str(contract[key]).strip().lower().startswith(("required", "unverified")):
            raise ValueError("Unresolved source evidence declaration: " + key)
    if contract["alert_time_semantics"] not in {"alarm_initiation_with_retrospective_classification", "alert_timestamp_is_severity_recorded_at"}:
        raise ValueError("Explicit alarm timing and retrospective outcome declaration required")
    if contract["steps_semantics"] not in {"increments", "cumulative_counter"}:
        raise ValueError("Unverified Steps semantics: supply documented increments or cumulative_counter")
    if contract["steps_semantics"] == "cumulative_counter":
        if contract.get("counter_reset_policy") not in {"daily", "decrease_only"}:
            raise ValueError("Cumulative counters require an explicit counter_reset_policy")
        if contract["counter_reset_policy"] == "daily":
            ZoneInfo(contract["source_timezone"])
    if contract["measurement_availability"] not in {"timestamp_is_available_at", "explicit_available_at"}:
        raise ValueError("Unknown measurement availability contract")
    if contract["clinical_history_policy"] not in {"dated_snapshots", "verified_baseline", "exclude"}:
        raise ValueError("Unknown clinical history policy")
    if contract["clinical_history_policy"] == "verified_baseline" and not contract.get("baseline_evidence"):
        raise ValueError("Baseline attributes require an explicit provenance declaration")


def validate_coverage(coverage):
    if "support_start" in coverage:
        if coverage.get("coverage_basis") != "researcher_attested_complete_alerts_with_observed_support":
            raise ValueError("Observed support requires an explicit retrospective coverage basis")
        coverage = dict(coverage, enrollment_time=coverage["support_start"])
    keys = ("enrollment_time", "measurement_coverage_end", "outcome_coverage_start", "outcome_coverage_end")
    times = {k: as_time(coverage[k]) for k in keys}
    if times["measurement_coverage_end"] <= times["enrollment_time"]:
        raise ValueError("Empty measurement observation interval")
    if times["outcome_coverage_end"] < times["outcome_coverage_start"]:
        raise ValueError("Invalid outcome coverage interval")
    return times


def build_features(measurements, coverage, histories, task, contract):
    """One patient at a time; no alerts or outcomes accepted by this function.

    timestamp is bucket start; available_at is its exclusive end. Delayed readings
    are conservatively excluded, never backfilled into a closed historical bucket.
    """
    validate_source_contract(contract)
    times = validate_coverage(coverage)
    sid = str(coverage["senior_id"])
    width = timedelta(minutes=task.grid_minutes)
    buckets = defaultdict(lambda: defaultdict(list))
    grouped = defaultdict(list)
    for source in measurements:
        if str(source["senior_id"]) != sid:
            raise ValueError("Cross-patient measurement in feature generation")
        at = as_time(source["date"])
        if not times["enrollment_time"] <= at < times["measurement_coverage_end"]:
            continue
        available = as_time(source["available_at"]) if contract["measurement_availability"] == "explicit_available_at" else at
        if available < at:
            raise ValueError("Measurement availability precedes acquisition")
        if available >= grid_floor(at, task) + width:
            continue
        grouped[(at, source["type"])].append(source)
    previous_counter, previous_counter_time = None, None
    for (at, kind), duplicates in sorted(grouped.items()):
        clean = [clean_measurement(r, task) for r in duplicates]
        combined = {}
        for name in clean[0]:
            vals = [r[name] for r in clean if r[name] is not None]
            if name == "steps" and len(set(vals)) > 1:
                if task.values.get('steps_duplicate_conflict_policy','reject')=='reject':
                    raise ValueError("Conflicting step values at identical timestamp; resolve source duplicates")
                vals=[]  # Ambiguous source snapshot is unavailable, never averaged.
            combined[name] = statistics.mean(vals) if vals else None
        # PP above averages only valid raw pairs, never independently combined components.
        if kind == "Steps" and combined["steps"] is not None:
            buckets[grid_floor(at, task)]["steps_source_value"].append((at, combined["steps"]))
        if kind == "Steps" and contract["steps_semantics"] == "cumulative_counter":
            current = combined["steps"]
            combined["steps"] = None
            if current is not None:
                day_reset = (previous_counter_time is not None and contract["counter_reset_policy"] == "daily"
                             and at.astimezone(ZoneInfo(contract["source_timezone"])).date()
                             != previous_counter_time.astimezone(ZoneInfo(contract["source_timezone"])).date())
                if previous_counter is not None and current >= previous_counter and not day_reset:
                    combined["steps"] = valid_value("steps", current - previous_counter, task)
                    buckets[grid_floor(at, task)]["steps_delta_interval_minutes"].append((at, (at-previous_counter_time).total_seconds()/60))
                if previous_counter is not None and (current < previous_counter or day_reset):
                    buckets[grid_floor(at, task)]["steps_counter_reset"].append((at, 1))
                # First reading/reset is unknown activity, not zero; starts next interval.
                previous_counter = current
                previous_counter_time = at
        for name, value in combined.items():
            if value is not None:
                buckets[grid_floor(at, task)][name].append((at, value))
    history = []
    if contract["clinical_history_policy"] != "exclude":
        for record in histories:
            if str(record["senior_id"]) != sid:
                raise ValueError("Cross-patient clinical snapshot")
            available = as_time(record["available_at"])
            if contract["clinical_history_policy"] == "verified_baseline" and available > times["enrollment_time"]:
                raise ValueError("Claimed baseline was recorded after enrollment")
            if set(record["values"]) - set(STATIC):
                raise ValueError("Unknown static feature")
            history.append((available, record["values"]))
    history.sort(key=lambda x: x[0])
    cursor, state, recency, recent = 0, {}, {}, deque()
    start = grid_floor(times["enrollment_time"], task)
    if start < times["enrollment_time"]:
        start += width
    while start + width <= times["measurement_coverage_end"]:
        end = start + width
        while cursor < len(history) and history[cursor][0] < end:
            state.update(history[cursor][1])
            cursor += 1
        row = {"senior_id": sid, "timestamp": iso(start), "available_at": iso(end)}
        for name in PHYSIOLOGY:
            readings = buckets.get(start, {}).get(name, [])
            values = [v for _, v in readings]
            row[name] = (sum(values) if name == "steps" else statistics.mean(values)) if values else None
            row["observed_" + name] = int(bool(values))
            if readings:
                recency[name] = max(at for at, _ in readings)
            row["time_since_last_" + name] = (end - recency[name]).total_seconds() / 60 if name in recency else None
        # PP is the mean of valid, deduplicated paired-reading differences above.
        # Do not recreate invalid PP from independently averaged bucket SBP/DBP.
        step_source = buckets.get(start, {}).get("steps_source_value", [])
        intervals = buckets.get(start, {}).get("steps_delta_interval_minutes", [])
        row["steps_source_value"] = step_source[-1][1] if step_source else None
        row["observed_steps_source"] = int(bool(step_source))
        row["steps_counter_reset"] = int(bool(buckets.get(start, {}).get("steps_counter_reset", [])))
        row["steps_delta_interval_minutes"] = max(v for _, v in intervals) if intervals else None
        row["shock_index"] = row["heartrate"] / row["sbp"] if row["heartrate"] is not None and row["sbp"] is not None else None
        recent.append(row)
        max_history = max(task.hr_sd_minutes, task.bp_trend_minutes, task.steps_sum_minutes)
        while recent and as_time(recent[0]["timestamp"]) < end - timedelta(minutes=max_history):
            recent.popleft()
        def observed(name, minutes):
            return [(as_time(r["timestamp"]), r[name]) for r in recent if as_time(r["timestamp"]) >= end - timedelta(minutes=minutes) and r[name] is not None]
        hr = [v for _, v in observed("heartrate", task.hr_sd_minutes)]
        row["hr_bucket_sd_4h"] = statistics.stdev(hr) if len(hr) >= 2 else None
        bp = observed("sbp", task.bp_trend_minutes)
        row["bp_trend_mmhg_per_hour"] = None
        if len(bp) >= 2:
            xs = [(at - bp[0][0]).total_seconds() / 3600 for at, _ in bp]
            ys = [v for _, v in bp]
            xm, ym = statistics.mean(xs), statistics.mean(ys)
            denom = sum((x - xm) ** 2 for x in xs)
            if denom:
                row["bp_trend_mmhg_per_hour"] = sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / denom
        steps = observed("steps", task.steps_sum_minutes)
        row["steps_sum_6h"] = sum(v for _, v in steps) if steps else None
        row["steps_observed_buckets_6h"] = len(steps)
        hour = start.hour + start.minute / 60
        row.update(hour_sin=math.sin(2 * math.pi * hour / 24), hour_cos=math.cos(2 * math.pi * hour / 24), is_night=int(hour < 6))
        for name in STATIC:
            value = state.get(name)
            if value is not None and not finite(value):
                raise ValueError("Clinical features must be numeric or NULL")
            row[name] = float(value) if value is not None else None
            row["known_" + name] = int(value is not None)
        yield row
        start = end
