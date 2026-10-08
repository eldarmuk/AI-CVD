"""Raw exports -> quality controls -> causal features -> indexed research datasets."""

import csv
import hashlib
import json
import platform
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .dataset import patient_split, sample_id
from .episodes import as_time, build_episodes, iso
from .features import (
    FEATURE_NAMES,
    PHYSIOLOGY,
    build_features,
    clean_measurement,
    validate_coverage,
    validate_source_contract,
)
from .index import window_index
from .storage import connect, digest, ingest_csv, patient_measurements, validate_raw, write_parquet
from .task import load_task
from .vector import vector_features


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def implementation_digest():
    return hashlib.sha256(
        "".join(digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))).encode()
    ).hexdigest()


def safe_artifact(root, name):
    candidate = (root / name).resolve()
    if not candidate.is_relative_to(root.resolve()) or candidate.is_symlink():
        raise ValueError("Artifact path escapes run directory")
    return candidate


def verify(run):
    run = Path(run)
    record = json.loads((run / "run.json").read_text(encoding="utf-8"))
    if record["status"] != "complete":
        raise ValueError("Run is incomplete")
    for name, expected in record["artifacts"].items():
        if digest(safe_artifact(run, name)) != expected:
            raise ValueError("Artifact integrity check failed")
    return record


def vector_frame(rows, coverage, task, contract):
    """Replace the private SQLite reader with canonical in-memory paired records."""
    if any(r["date"] != r["available_at"] for r in rows):
        raise ValueError("Immediate-availability declaration contradicts records")
    normalized = [
        {"date": r["date"], "type": r["type"], **clean_measurement(r, task)} for r in rows
    ]
    frame = pd.DataFrame(normalized).reindex(columns=["date", "type", *PHYSIOLOGY])
    groups = frame.groupby(["date", "type"], sort=True)
    extremes = groups.steps.agg(["min", "max"])
    conflicts = extremes["min"].notna() & (extremes["min"] != extremes["max"])
    if conflicts.any() and task.values.get("steps_duplicate_conflict_policy", "reject") == "reject":
        raise ValueError("Conflicting step values at identical timestamp")
    clean = groups[list(PHYSIOLOGY)].mean()
    clean.loc[conflicts, "steps"] = np.nan
    features = vector_features(clean.reset_index(), coverage, task, contract)
    features.insert(0, "timestamp", [iso(at) for at in features.index])
    features.insert(
        1,
        "available_at",
        [iso(at + pd.Timedelta(minutes=task.grid_minutes)) for at in features.index],
    )
    features.insert(2, "senior_id", coverage["senior_id"])
    return features.reset_index(drop=True)


def run_pipeline(
    inputs, output, *, resume=False, engine="reference", threads=2, memory_limit="512MB", task=None
):
    inputs, output = Path(inputs).resolve(), Path(output).resolve()
    task = task or load_task()
    if engine not in {"reference", "vector"}:
        raise ValueError("Unknown processing engine")
    source_paths = [
        inputs / n for n in ("telemetry.csv", "coverage.csv", "events.csv", "source.json")
    ]
    source_hashes = {p.name: digest(p) for p in source_paths}
    identity = {
        "inputs": source_hashes,
        "task_identifier": task.identifier,
        "implementation_sha256": implementation_digest(),
        "engine": engine,
    }
    if output == inputs or inputs.is_relative_to(output):
        raise ValueError("Output must not contain the input directory")
    if output.exists():
        if not resume or not (output / "run.json").is_file():
            raise FileExistsError("Choose a new output directory or resume a verified pipeline run")
        previous = json.loads((output / "run.json").read_text())
        if previous["identity"] != identity:
            raise ValueError("Inputs, implementation, task or engine changed; start a new run")
        if previous["status"] == "complete":
            return verify(output)
    else:
        output.mkdir(parents=True)
    lock = output / ".run.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write("exclusive pipeline writer\n")
    started = time.perf_counter()
    db = None
    state = {"identity": identity, "status": "running", "artifacts": {}}
    try:
        save_json(output / "run.json", state)
        source = json.loads((inputs / "source.json").read_text(encoding="utf-8"))
        contract = source["contract"]
        validate_source_contract(contract)
        if contract["clinical_history_policy"] != "exclude":
            raise ValueError("This public source adapter excludes clinical histories")
        if (
            engine == "vector"
            and contract["measurement_availability"] != "timestamp_is_available_at"
        ):
            raise ValueError(
                "Vector path requires immediate availability; use reference for late arrivals"
            )
        coverage_rows, alerts = read_csv(inputs / "coverage.csv"), read_csv(inputs / "events.csv")
        subjects = [r["subject_id"] for r in coverage_rows]
        if len(set(subjects)) != len(subjects) or not subjects or any(not sid for sid in subjects):
            raise ValueError("Unique nonempty coverage identities required")
        if any(a["subject_id"] not in subjects for a in alerts):
            raise ValueError("Event lacks a coverage record")
        episodes = build_episodes(alerts)
        db = connect(output / "telemetry.duckdb", threads=threads, memory_limit=memory_limit)
        ingest_csv(db, inputs / "telemetry.csv")
        source_rows = db.execute("SELECT count(*) FROM raw_measurements").fetchone()[0]
        quality = {"structural": validate_raw(db), "measurement": Counter()}
        known = {
            r[0]
            for r in db.execute(
                "SELECT DISTINCT subject_id FROM validated WHERE rejection_reason IS NULL"
            ).fetchall()
        }
        if known - set(subjects):
            raise ValueError("Measurement lacks an explicit coverage contract")
        (output / "patients").mkdir(exist_ok=True)
        patient_index, flow = [], Counter()
        stages = {"ingestion_validation_seconds": time.perf_counter() - started}
        feature_start = time.perf_counter()
        for coverage in coverage_rows:
            sid = coverage["subject_id"]
            cov = {
                "senior_id": sid,
                "enrollment_time": coverage["observation_start"],
                "measurement_coverage_end": coverage["observation_end"],
                "outcome_coverage_start": coverage["outcome_start"],
                "outcome_coverage_end": coverage["outcome_end"],
            }
            validate_coverage(cov)
            own_episodes = [e for e in episodes if e["senior_id"] == sid]
            if any(
                not as_time(cov["outcome_coverage_start"])
                <= as_time(m["timestamp"])
                <= as_time(cov["outcome_coverage_end"])
                for e in own_episodes
                for m in e["constituents"]
            ):
                raise ValueError("Recorded event outside outcome coverage")
            rows, counts = patient_measurements(db, sid, task)
            quality["measurement"].update(counts)
            if contract["measurement_availability"] == "timestamp_is_available_at" and any(
                r["date"] != r["available_at"] for r in rows
            ):
                raise ValueError("Availability declaration contradicts records")
            if not rows:
                raise ValueError("Coverage subject has no structurally valid telemetry")
            frame = (
                pd.DataFrame(build_features(rows, cov, [], task, contract))
                if engine == "reference"
                else vector_frame(rows, cov, task, contract)
            )
            if frame.empty:
                raise ValueError("Coverage contains no complete processing bucket")
            indexed = frame.set_index(pd.DatetimeIndex(pd.to_datetime(frame.timestamp, utc=True)))
            indices, counts, events = window_index(indexed, own_episodes, cov, task)
            flow.update(counts)
            patient_key = hashlib.sha256(sid.encode()).hexdigest()
            parquet = output / "patients" / f"{patient_key}.parquet"
            write_parquet(db, frame, parquet)
            arrays = output / "patients" / f"{patient_key}.npz"
            np.savez_compressed(
                arrays, features=frame[list(FEATURE_NAMES)].to_numpy(dtype=np.float32), **indices
            )
            ends = indices["end_rows"]
            ids = [sample_id(sid, frame.iloc[int(e)]["available_at"]) for e in ends]
            patient_index.append(
                {
                    "subject_id": sid,
                    "split": patient_split(sid, task),
                    "features": parquet.relative_to(output).as_posix(),
                    "arrays": arrays.relative_to(output).as_posix(),
                    "feature_rows": len(frame),
                    "windows": len(ends),
                    "sample_ids": ids,
                    "events": events,
                    "grid_start": frame.iloc[0]["timestamp"],
                    "coverage": cov,
                }
            )
        stages["features_windows_storage_seconds"] = time.perf_counter() - feature_start
        save_json(output / "patients.json", patient_index)
        save_json(output / "episodes.json", episodes)
        save_json(output / "quality.json", quality)
        save_json(output / "cohort_flow.json", dict(flow))
        # Counts reconcile disjoint eligibility decisions; events can have multiple loss reasons.
        excluded = sum(v for k, v in flow.items() if k.startswith("excluded_"))
        if flow["candidate_prediction_times"] != excluded + flow["eligible_prediction_times"]:
            raise ValueError("Cohort flow failed reconciliation")
        db.execute("CHECKPOINT")
        db.close()
        db = None
        stages["total_seconds"] = time.perf_counter() - started
        save_json(
            output / "performance.json",
            {
                "stages": stages,
                "input_rows": source_rows,
                "input_rows_per_second": source_rows / stages["total_seconds"],
                "subjects": len(subjects),
                "feature_rows": sum(p["feature_rows"] for p in patient_index),
                "engine": engine,
                "threads": threads,
                "duckdb_memory_limit": memory_limit,
                "python": platform.python_version(),
                "platform": platform.system(),
                "memory_note": "DuckDB budget is not a process peak-RSS measurement.",
            },
        )
        state.update(
            status="complete",
            synthetic=source.get("synthetic") is True,
            feature_names=list(FEATURE_NAMES),
            task_values=task.values,
            subjects=len(subjects),
            raw_rows=source_rows,
            engine=engine,
        )
        artifact_paths = [
            p for p in output.rglob("*") if p.is_file() and p.name not in {"run.json", ".run.lock"}
        ]
        state["artifacts"] = {p.relative_to(output).as_posix(): digest(p) for p in artifact_paths}
        save_json(output / "run.json", state)
        return verify(output)
    except Exception:
        state["status"] = "incomplete"
        save_json(output / "run.json", state)
        raise
    finally:
        if db is not None:
            db.close()
        lock.unlink(missing_ok=True)


def sequence_batches(run, split, batch_size=64, training_only=False):
    """Verify artifacts, then lazily materialize windows with stable sample identities."""
    if split not in {"train", "validation", "test"} or batch_size < 1:
        raise ValueError("Valid split and positive batch size required")
    if training_only and split != "train":
        raise ValueError("Training-only selection cannot use held-out patients")
    run = Path(run)
    metadata = verify(run)
    length = metadata["task_values"]["sequence_steps"]
    for patient in json.loads((run / "patients.json").read_text()):
        if patient["split"] != split:
            continue
        with np.load(safe_artifact(run, patient["arrays"]), allow_pickle=False) as archive:
            selected = (
                np.flatnonzero(archive["training"])
                if training_only
                else np.arange(len(archive["end_rows"]))
            )
            for offset in range(0, len(selected), batch_size):
                chosen = selected[offset : offset + batch_size]
                ends = archive["end_rows"][chosen]
                rows = ends[:, None] - length + 1 + np.arange(length)
                if (rows < 0).any() or (rows >= len(archive["features"])).any():
                    raise ValueError("Invalid sequence bounds")
                yield {
                    "X": archive["features"][rows],
                    "y": archive["target"][chosen],
                    "sample_ids": [patient["sample_ids"][int(i)] for i in chosen],
                    "subject_id": patient["subject_id"],
                    "split": split,
                }
