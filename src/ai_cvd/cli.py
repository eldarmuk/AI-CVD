"""Private-data ingestion and public synthetic reproduction; never trains a model."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import tomllib
from datetime import datetime
from zoneinfo import ZoneInfo

from .task import load_task, DEFAULT_TASK
from .episodes import as_time, iso, build_episodes, write_episode_tables
from .features import FEATURE_NAMES, build_features, validate_source_contract
from .dataset import generate_patient_manifest, training_manifest, development_manifest, patient_split


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, rows):
    with open(path, "x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + "\n")


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def localize(value, contract):
    """Naive source times require a declared timezone; reject ambiguous DST times."""
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        if not contract.get("source_timezone"):
            raise ValueError("Naive source timestamp requires source_timezone declaration")
        zone = ZoneInfo(contract["source_timezone"])
        one, two = parsed.replace(tzinfo=zone, fold=0), parsed.replace(tzinfo=zone, fold=1)
        if one.utcoffset() != two.utcoffset():
            raise ValueError("Ambiguous/nonexistent local source timestamp: resolve DST before extraction")
        parsed = one
    return iso(parsed)


class SQLiteSource:
    def __init__(self, path, contract):
        self.path, self.contract = Path(path), contract
        self.connection = sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA query_only=ON")
        self.connection.execute("BEGIN")
        self.validate_raw_schema()

    def validate_raw_schema(self):
        # Historical processed tables have already lost clipped values/burst members.
        columns = {r[1] for r in self.connection.execute("PRAGMA table_info('alerts')").fetchall()}
        if 'severity' in columns or not {'alert_id', 'senior_id', 'alert_date', 'sos_note'} <= columns:
            self.connection.close()
            raise ValueError("Use the uncompressed raw alert source, not a processed database")

    def patients(self):
        return [str(r[0]) for r in self.connection.execute("SELECT id FROM seniors ORDER BY id")]

    def measurements(self, sid):
        rows = self.connection.execute("SELECT * FROM measurements WHERE senior_id=? ORDER BY date,id", (sid,))
        for source in rows:
            row = dict(source)
            row["date"] = localize(row["date"], self.contract)
            if self.contract["measurement_availability"] == "explicit_available_at":
                row["available_at"] = localize(row["available_at"], self.contract)
            yield row

    def alerts(self, sid):
        for source in self.connection.execute("SELECT * FROM alerts WHERE senior_id=? ORDER BY alert_date,alert_id", (sid,)):
            row = dict(source)
            row["alert_date"] = localize(row["alert_date"], self.contract)
            # Classify from raw notes; never reuse first-alert-only processed severity.
            row.pop("severity", None)
            yield row

    def close(self):
        self.connection.close()


class DuckDBSource(SQLiteSource):
    """DuckDB pushdown read adapter; feature semantics remain shared Python code.

    Requires the sqlite extension to have been provisioned. No network install,
    raw-table mutation, or divergent SQL-derived clinical features.
    """
    def __init__(self, path, contract):
        import duckdb
        self.path, self.contract = Path(path), contract
        self.connection = duckdb.connect()
        extension_dir = contract.get("duckdb_extension_directory")
        if extension_dir:
            escaped_dir = str(Path(extension_dir).resolve()).replace("'", "''")
            self.connection.execute(f"SET extension_directory='{escaped_dir}'")
        self.connection.execute("LOAD sqlite")
        # SQLite affinity is permissive. Preserve raw strings (including offsets and
        # invalid numeric sentinels) for the shared parser instead of inferred casts.
        self.connection.execute("SET sqlite_all_varchar=true")
        escaped = str(self.path.resolve()).replace("'", "''")
        self.connection.execute(f"ATTACH '{escaped}' AS source (TYPE sqlite, READ_ONLY)")
        self.connection.execute("USE source")
        self.validate_raw_schema()

    def patients(self):
        return [str(r[0]) for r in self.connection.execute("SELECT id FROM seniors ORDER BY id").fetchall()]

    def records(self, query, sid):
        cursor = self.connection.execute(query, [sid])
        names = [x[0] for x in cursor.description]
        while True:
            block = cursor.fetchmany(10000)
            if not block:
                return
            for row in block:
                yield dict(zip(names, row))

    def measurements(self, sid):
        for row in self.records("SELECT * FROM measurements WHERE senior_id=? ORDER BY date,id", sid):
            row["date"] = localize(row["date"], self.contract)
            if self.contract["measurement_availability"] == "explicit_available_at":
                row["available_at"] = localize(row["available_at"], self.contract)
            yield row

    def alerts(self, sid):
        for row in self.records("SELECT * FROM alerts WHERE senior_id=? ORDER BY alert_date,alert_id", sid):
            row["alert_date"] = localize(row["alert_date"], self.contract)
            row.pop("severity", None)
            yield row


def build_run(source, coverages, histories, task, contract, output, endpoint=None, training_policy=None, source_snapshot_id=None):
    validate_source_contract(contract)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)  # Old results can never be overwritten.
    if not source_snapshot_id or not str(source_snapshot_id).strip():
        raise ValueError("Explicit immutable source snapshot identifier required")
    if endpoint is not None and endpoint not in {task.endpoint, task.secondary_endpoint}:
        raise ValueError("Unknown endpoint")
    if training_policy is not None and training_policy not in {"event_free_windows", "never_event_patients"}:
        raise ValueError("Unknown training policy")
    by_patient = {}
    for c in coverages:
        sid = str(c["senior_id"])
        if sid in by_patient:
            raise ValueError("Duplicate coverage declaration")
        by_patient[sid] = c
    history_map = {}
    histories = list(histories)
    for h in histories:
        history_map.setdefault(str(h["senior_id"]), []).append(h)
    patients = source.patients()
    if set(by_patient) - set(patients):
        raise ValueError("Coverage refers to patients absent from source")
    if set(history_map) - set(patients):
        raise ValueError("History refers to patients absent from source")
    # Split assignment does not inspect labels, density or eligible windows.
    write_jsonl(output / "patient_splits.jsonl", ({"senior_id": sid, "split": patient_split(sid, task)} for sid in patients))
    write_jsonl(output / "coverage_declarations.jsonl", by_patient.values())
    write_jsonl(output / "clinical_snapshots.jsonl", histories)
    cohort = Counter(patients_initially_available=len(patients))
    episode_db = sqlite3.connect(output / "episodes.sqlite")
    write_episode_tables(episode_db, [])
    episode_db.commit()
    paths = ["features.jsonl", "episodes.jsonl", "stream_manifest.jsonl", "unsupervised_training_manifest.jsonl"]
    handles = [open(output / name, "x", encoding="utf-8") for name in paths]
    def emit(handle, row):
        handle.write(json.dumps(row, allow_nan=False) + "\n")
    try:
        for sid in patients:
            if sid not in by_patient:
                cohort["patients_excluded_missing_coverage"] += 1
                continue
            coverage = by_patient[sid]
            episodes = build_episodes(source.alerts(sid), task)
            features = list(build_features(source.measurements(sid), coverage, history_map.get(sid, []), task, contract))
            stream, flow = generate_patient_manifest(features, episodes, coverage, task, endpoint)
            flow.pop("patients_initially_available")
            cohort.update(flow)
            train = training_manifest(stream, episodes, task, training_policy, coverage)
            cohort["unsupervised_training_windows"] += len(train)
            for handle, rows in zip(handles, [features, episodes, stream, train]):
                for row in rows:
                    emit(handle, row)
            for e in episodes:
                keys = ["episode_id", "senior_id", "episode_start", "episode_end", "first_severity", "maximum_severity", "final_recorded_severity", "escalation_recorded_at", "maximum_severity_recorded_at", "constituent_count", "outcome_basis"]
                episode_db.execute("INSERT INTO alert_episodes_v2 VALUES (?,?,?,?,?,?,?,?,?,?,?)", [e[k] for k in keys])
                episode_db.executemany("INSERT INTO alert_episode_members_v2 VALUES (?,?,?,?)", [(e["episode_id"], a["alert_id"], a["timestamp"], a["severity"]) for a in e["constituents"]])
            episode_db.commit()
    finally:
        for handle in handles:
            handle.close()
        episode_db.close()
    with open(output / "cohort_flow.json", "x", encoding="utf-8") as handle:
        json.dump({"counts": dict(cohort), "window_exclusions": "sequential/disjoint", "patient_event_reason_counts": "may overlap; event loss means no eligible positive prediction", "unassessable_events": "Patients without authoritative coverage are excluded; their event opportunity loss is not estimable."}, handle, indent=2)
    artifacts = {p.name: file_hash(p) for p in output.iterdir() if p.is_file()}
    implementation = hashlib.sha256()
    for code in sorted(Path(__file__).parent.glob("*.py")):
        implementation.update(code.name.encode())
        implementation.update(code.read_bytes())
    metadata = {"status": "complete", "task_identifier": task.identifier, "task": task.values,
                "endpoint": endpoint or task.endpoint, "training_policy": training_policy or task.training_policy,
                "feature_names": FEATURE_NAMES, "source_contract": contract,
                "source_snapshot_id": source_snapshot_id, "artifacts_sha256": artifacts,
                "runtime": sys.version, "source_backend": type(source).__name__,
                "implementation_sha256": implementation.hexdigest(), "sampling": "complete_eligible_stream"}
    with open(output / "run_metadata.json", "x", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return metadata


def verify_run(path, task):
    path = Path(path)
    with open(path / "run_metadata.json", encoding="utf-8") as handle:
        meta = json.load(handle)
    if meta["status"] != "complete" or meta["task_identifier"] != task.identifier or tuple(meta["feature_names"]) != FEATURE_NAMES:
        raise ValueError("Incomplete, legacy, or mismatched task/schema")
    required = {"features.jsonl", "episodes.jsonl", "episodes.sqlite", "stream_manifest.jsonl",
                "unsupervised_training_manifest.jsonl", "cohort_flow.json", "patient_splits.jsonl",
                "coverage_declarations.jsonl", "clinical_snapshots.jsonl"}
    if not required <= set(meta["artifacts_sha256"]):
        raise ValueError("Missing required artifact checksum")
    for name, digest in meta["artifacts_sha256"].items():
        if Path(name).name != name:
            raise ValueError("Invalid artifact path")
        if file_hash(path / name) != digest:
            raise ValueError("Artifact checksum mismatch")
    return meta


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, default=DEFAULT_TASK)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--raw-db", required=True, type=Path)
    build.add_argument("--source-contract", required=True, type=Path)
    build.add_argument("--coverage", required=True, type=Path, help="CSV with externally established observation/outcome intervals")
    build.add_argument("--history", type=Path, help="Dated numeric clinical snapshots JSONL; never inferred from undated tables")
    build.add_argument("--output", required=True, type=Path)
    build.add_argument("--source-snapshot-id", required=True)
    build.add_argument("--backend", choices=["sqlite", "duckdb"], default="sqlite")
    build.add_argument("--endpoint", choices=["level3", "level2_or_level3"], default="level3")
    build.add_argument("--training-policy", choices=["event_free_windows", "never_event_patients"])
    demo = commands.add_parser("synthetic")
    demo.add_argument("--output", required=True, type=Path)
    development = commands.add_parser("development")
    development.add_argument("--run", required=True, type=Path)
    development.add_argument("--output", required=True, type=Path)
    development.add_argument("--per-class", type=int, default=1000)
    export = commands.add_parser("sequences")
    export.add_argument("--run", required=True, type=Path)
    export.add_argument("--output", required=True, type=Path)
    export.add_argument("--split", required=True, choices=["train", "validation", "test"])
    export.add_argument("--kind", choices=["stream", "unsupervised_training"], default="stream")
    args = parser.parse_args(argv)
    task = load_task(args.task)
    print(f"Task: {task.identifier}; {task.sequence_steps} x {task.grid_minutes} min = {task.lookback_minutes} min; horizon {task.horizon_minutes} min")
    if args.command == "synthetic":
        from .synthetic import fixture
        source, coverage, histories, contract = fixture(task)
        build_run(source, coverage, histories, task, contract, args.output, source_snapshot_id="fictional-fixture-v2")
    elif args.command == "sequences":
        from .arrays import export_sequences
        export_sequences(args.run, args.output, task, args.split, args.kind)
    elif args.command == "development":
        meta = verify_run(args.run, task)
        # Optional development extraction is separate; no test-based selection/calibration.
        selected = development_manifest(read_jsonl(args.run / "stream_manifest.jsonl"), args.per_class, task.split_seed)
        args.output.mkdir(parents=True, exist_ok=False)
        write_jsonl(args.output / "development_manifest.jsonl", selected)
        with open(args.output / "metadata.json", "x", encoding="utf-8") as handle:
            json.dump({"task_identifier": task.identifier, "endpoint": meta["endpoint"], "sampling": "balanced_development_not_natural_prevalence", "source_manifest_sha256": meta["artifacts_sha256"]["stream_manifest.jsonl"], "per_class": args.per_class, "seed": task.split_seed}, handle, indent=2)
    else:
        if args.source_contract.suffix == ".toml":
            with open(args.source_contract, "rb") as handle:
                contract = tomllib.load(handle)
        else:
            with open(args.source_contract, encoding="utf-8") as handle:
                contract = json.load(handle)
        validate_source_contract(contract)
        with open(args.coverage, newline="", encoding="utf-8") as handle:
            coverage = list(csv.DictReader(handle))
        histories = list(read_jsonl(args.history)) if args.history else []
        if contract["clinical_history_policy"] != "exclude" and not args.history:
            raise ValueError("Clinical-history policy requires a dated history file")
        source = (SQLiteSource if args.backend == "sqlite" else DuckDBSource)(args.raw_db, contract)
        try:
            build_run(source, coverage, histories, task, contract, args.output, args.endpoint, args.training_policy, args.source_snapshot_id)
        finally:
            source.close()
    print("Completed without training. Existing artifacts were not overwritten.")


if __name__ == "__main__":
    main()
