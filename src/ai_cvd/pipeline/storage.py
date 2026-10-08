"""Transactional canonical CSV ingestion into a local analytical DuckDB database."""

import csv
import hashlib
import re
from pathlib import Path

import duckdb
import pandas as pd

RAW_COLUMNS = (
    "record_id",
    "subject_id",
    "timestamp",
    "available_at",
    "measurement",
    "value",
    "sbp",
    "dbp",
    "unit",
    "device_status",
)
UNITS = {
    "Heartrate": "bpm",
    "Temperature": "degC",
    "BloodPressure": "mmHg",
    "Saturation": "%",
    "Steps": "count",
}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def connect(path, *, threads=2, memory_limit="512MB", read_only=False):
    if (
        type(threads) is not int
        or threads < 1
        or not re.fullmatch(r"[1-9][0-9]*(?:MB|GB)", memory_limit)
    ):
        raise ValueError("Positive threads and an explicit MB/GB memory budget required")
    db = duckdb.connect(str(path), read_only=read_only)
    db.execute("SET TimeZone = 'UTC'")
    db.execute("SET threads = ?", [threads])
    db.execute("SET memory_limit = ?", [memory_limit])
    if not read_only:
        db.execute(
            "CREATE TABLE IF NOT EXISTS ingested_files (digest VARCHAR PRIMARY KEY, row_count BIGINT)"
        )
        db.execute("""CREATE TABLE IF NOT EXISTS raw_measurements (
            source_digest VARCHAR, record_id VARCHAR PRIMARY KEY, subject_id VARCHAR,
            timestamp VARCHAR, available_at VARCHAR, measurement VARCHAR, value VARCHAR,
            sbp VARCHAR, dbp VARCHAR, unit VARCHAR, device_status VARCHAR)""")
    return db


def ingest_csv(db, path):
    """All rows from one content-addressed file commit together; replay is idempotent."""
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as stream:
        if tuple(next(csv.reader(stream), [])) != RAW_COLUMNS:
            raise ValueError("Canonical telemetry header differs; use an explicit source adapter")
    fingerprint = digest(path)
    prior = db.execute(
        "SELECT row_count FROM ingested_files WHERE digest=?", [fingerprint]
    ).fetchone()
    if prior:
        return {"inserted": 0, "source_rows": prior[0], "replayed": True}
    db.read_csv(str(path), header=True, all_varchar=True).create_view("incoming", replace=True)
    db.execute("BEGIN TRANSACTION")
    try:
        count = db.execute("SELECT count(*) FROM incoming").fetchone()[0]
        if db.execute(
            "SELECT count(*) FROM incoming WHERE record_id IS NULL OR record_id='' "
        ).fetchone()[0]:
            raise ValueError("Missing record identity")
        if db.execute("SELECT count(*)-count(DISTINCT record_id) FROM incoming").fetchone()[0]:
            raise ValueError("Duplicate record identity within source file")
        if db.execute(
            "SELECT count(*) FROM incoming JOIN raw_measurements USING(record_id)"
        ).fetchone()[0]:
            raise ValueError("Record identity reused across distinct source files")
        db.execute("INSERT INTO raw_measurements SELECT ?, * FROM incoming", [fingerprint])
        db.execute("INSERT INTO ingested_files VALUES (?, ?)", [fingerprint, count])
        db.execute("COMMIT")
    except Exception:
        db.execute("ROLLBACK")
        raise
    finally:
        db.execute("DROP VIEW incoming")
    return {"inserted": count, "source_rows": count, "replayed": False}


def validate_raw(db):
    """Quarantine malformed records; retain reason codes rather than guessing semantics."""
    db.execute("""CREATE OR REPLACE TABLE normalized AS SELECT *,
        try_cast(timestamp AS TIMESTAMPTZ) AS acquired,
        try_cast(available_at AS TIMESTAMPTZ) AS received,
        try_cast(value AS DOUBLE) AS numeric_value,
        try_cast(sbp AS DOUBLE) AS numeric_sbp,
        try_cast(dbp AS DOUBLE) AS numeric_dbp
        FROM raw_measurements""")
    db.execute("""CREATE OR REPLACE TABLE validated AS SELECT *, CASE
        WHEN subject_id IS NULL OR subject_id='' THEN 'missing_subject'
        WHEN acquired IS NULL OR received IS NULL
             OR NOT regexp_matches(timestamp, '(Z|[+-][0-9]{2}:[0-9]{2})$')
             OR NOT regexp_matches(available_at, '(Z|[+-][0-9]{2}:[0-9]{2})$') THEN 'invalid_timestamp'
        WHEN received < acquired THEN 'availability_before_acquisition'
        WHEN measurement NOT IN ('Heartrate','Temperature','BloodPressure','Saturation','Steps')
             OR measurement IS NULL THEN 'unknown_measurement'
        WHEN unit IS NULL OR NOT ((measurement='Heartrate' AND unit='bpm')
             OR (measurement='Temperature' AND unit='degC')
             OR (measurement='BloodPressure' AND unit='mmHg')
             OR (measurement='Saturation' AND unit='%')
             OR (measurement='Steps' AND unit='count')) THEN 'unknown_unit'
        WHEN device_status IS NULL OR device_status NOT IN ('ok','off_body','device_error') THEN 'unknown_device_status'
        ELSE NULL END AS rejection_reason FROM normalized""")
    db.execute(
        "CREATE OR REPLACE VIEW quarantine AS SELECT * FROM validated WHERE rejection_reason IS NOT NULL"
    )
    return dict(
        db.execute(
            "SELECT coalesce(rejection_reason,'accepted'), count(*) FROM validated GROUP BY 1 ORDER BY 1"
        ).fetchall()
    )


def patient_measurements(db, subject, task):
    """Keep invalid values missing; preserve valid paired BP and explicit availability."""
    from .episodes import iso
    from .features import clean_measurement

    frame = db.execute(
        """SELECT acquired, received, measurement, numeric_value, numeric_sbp,
        numeric_dbp, device_status FROM validated WHERE subject_id=? AND rejection_reason IS NULL
        ORDER BY acquired, measurement, record_id""",
        [subject],
    ).fetchdf()
    rows, counters = [], {"device_flagged": 0, "invalid_values": 0, "late_arrivals": 0}
    for at, received, kind, value, sbp, dbp, status in frame.itertuples(index=False, name=None):
        row = {
            "senior_id": subject,
            "date": iso(at),
            "available_at": iso(received),
            "type": kind,
            "value": value,
            "sbp": sbp,
            "dbp": dbp,
        }
        if status != "ok":
            row.update(value=None, sbp=None, dbp=None)
            counters["device_flagged"] += 1
        cleaned = clean_measurement(row, task)
        counters["invalid_values"] += int(any(v is None for v in cleaned.values()))
        counters["late_arrivals"] += int(
            received
            >= at.floor(f"{task.grid_minutes}min") + pd.Timedelta(minutes=task.grid_minutes)
        )
        rows.append(row)
    return rows, counters


def write_parquet(db, frame, path):
    """Write through DuckDB; no Arrow dependency or private database extension."""
    db.register("export_frame", frame)
    try:
        db.sql("SELECT * FROM export_frame").write_parquet(str(path), compression="zstd")
    finally:
        db.unregister("export_frame")
