import csv
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from ai_cvd.pipeline.dataset import generate_patient_manifest, patient_split, sequence_for_sample
from ai_cvd.pipeline.episodes import build_episodes
from ai_cvd.pipeline.features import FEATURE_NAMES, build_features, clean_measurement
from ai_cvd.pipeline.fixture import generate_raw
from ai_cvd.pipeline.index import window_index
from ai_cvd.pipeline.runner import run_pipeline, sequence_batches, vector_frame, verify
from ai_cvd.pipeline.storage import RAW_COLUMNS, connect, ingest_csv, validate_raw
from ai_cvd.pipeline.task import load_task


def contract():
    return {
        "steps_semantics": "cumulative_counter",
        "counter_reset_policy": "daily",
        "source_timezone": "UTC",
        "steps_evidence": "fictional test",
        "measurement_availability": "timestamp_is_available_at",
        "availability_evidence": "fictional test",
        "coverage_evidence": "fictional test",
        "clinical_history_policy": "exclude",
        "alert_time_semantics": "alarm_initiation_with_retrospective_classification",
        "alert_time_evidence": "fictional test",
    }


def coverage():
    return {
        "senior_id": "FICTIONAL",
        "enrollment_time": "2099-01-01T00:00:00Z",
        "measurement_coverage_end": "2099-01-02T00:00:00Z",
        "outcome_coverage_start": "2099-01-01T00:00:00Z",
        "outcome_coverage_end": "2099-01-02T00:00:00Z",
    }


def measurements():
    rows = []
    for tick, at in enumerate(pd.date_range("2099-01-01T00:01:00Z", periods=144, freq="10min")):
        for kind, value in [
            ("Heartrate", 65 + tick % 7),
            ("BloodPressure", None),
            ("Steps", 8 * tick),
            ("Saturation", 96),
            ("Temperature", 36.5),
        ]:
            rows.append(
                {
                    "senior_id": "FICTIONAL",
                    "date": at.isoformat(),
                    "available_at": at.isoformat(),
                    "type": kind,
                    "value": value,
                    "sbp": 120,
                    "dbp": 75,
                }
            )
    return rows


def write_raw(path, rows):
    with path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=RAW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def raw_row(identifier="SYNTHETIC_1", **changes):
    return dict(
        {
            "record_id": identifier,
            "subject_id": "FICTIONAL",
            "timestamp": "2099-01-01T00:01:00Z",
            "available_at": "2099-01-01T00:01:05Z",
            "measurement": "Heartrate",
            "value": "72",
            "sbp": "",
            "dbp": "",
            "unit": "bpm",
            "device_status": "ok",
        },
        **changes,
    )


def test_transactional_ingestion_idempotence_and_rollback(tmp_path):
    path = tmp_path / "raw.csv"
    write_raw(path, [raw_row()])
    with connect(tmp_path / "test.duckdb") as db:
        assert ingest_csv(db, path)["inserted"] == 1
        assert ingest_csv(db, path)["replayed"]
        write_raw(path, [raw_row("SYNTHETIC_2"), raw_row("SYNTHETIC_2")])
        with pytest.raises(ValueError):
            ingest_csv(db, path)
        assert db.execute("SELECT count(*) FROM raw_measurements").fetchone()[0] == 1
        write_raw(path, [raw_row(value="90")])
        with pytest.raises(ValueError):
            ingest_csv(db, path)
        assert db.execute("SELECT count(*) FROM ingested_files").fetchone()[0] == 1


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"unit": "unknown"}, "unknown_unit"),
        ({"timestamp": "2099-01-01"}, "invalid_timestamp"),
        ({"available_at": "2098-01-01T00:00:00Z"}, "availability_before_acquisition"),
        ({"measurement": "unmapped"}, "unknown_measurement"),
    ],
)
def test_malformed_raw_is_quarantined(tmp_path, change, reason):
    path = tmp_path / "raw.csv"
    write_raw(path, [raw_row(**change)])
    with connect(tmp_path / "test.duckdb") as db:
        ingest_csv(db, path)
        assert validate_raw(db) == {reason: 1}


def test_wrong_schema_is_not_guessed(tmp_path):
    path = tmp_path / "raw.csv"
    path.write_text("arbitrary,columns\n1,2\n")
    with connect(tmp_path / "test.duckdb") as db, pytest.raises(ValueError):
        ingest_csv(db, path)


def test_invalid_values_remain_missing_and_bp_pair_is_not_repaired():
    task = load_task()
    assert clean_measurement({"type": "Heartrate", "value": 9999}, task) == {"heartrate": None}
    assert clean_measurement({"type": "BloodPressure", "sbp": 80, "dbp": 100}, task) == {
        "sbp": None,
        "dbp": None,
        "pulse_pressure": None,
    }


def test_delayed_arrival_and_future_reading_do_not_backfill():
    task, cov, spec = load_task(), coverage(), contract()
    spec["measurement_availability"] = "explicit_available_at"
    base = measurements()[:1]
    first = list(build_features(base, cov, [], task, spec))
    delayed = dict(base[0], value=999, available_at="2099-01-01T00:05:00Z")
    second = list(build_features(base + [delayed], cov, [], task, spec))
    assert first == second
    future = dict(
        base[0], date="2099-01-01T01:01:00Z", available_at="2099-01-01T01:01:00Z", value=180
    )
    changed = list(build_features(base + [future], cov, [], task, spec))
    assert changed[:12] == first[:12]


def test_counter_reset_unknown_not_zero_and_conflicting_duplicate_masked():
    spec, task, cov = contract(), load_task(), coverage()
    rows = [
        {"senior_id": "FICTIONAL", "type": "Steps", "date": at, "available_at": at, "value": value}
        for at, value in [
            ("2099-01-01T00:01:00Z", 100),
            ("2099-01-01T00:06:00Z", 110),
            ("2099-01-01T00:11:00Z", 4),
            ("2099-01-01T00:16:00Z", 9),
        ]
    ]
    features = list(build_features(rows, cov, [], task, spec))
    assert [r["steps"] for r in features[:4]] == [None, 10, None, 5]
    assert features[2]["steps_counter_reset"] == 1
    conflict = list(build_features(rows + [dict(rows[1], value=120)], cov, [], task, spec))
    assert conflict[1]["steps"] is None


def test_vector_reference_parity_and_continuous_window_index():
    task, spec, cov, rows = load_task(), contract(), coverage(), measurements()
    reference = pd.DataFrame(build_features(rows, cov, [], task, spec))
    vector = vector_frame(rows, cov, task, spec)
    np.testing.assert_allclose(
        reference[list(FEATURE_NAMES)].to_numpy(dtype=float),
        vector[list(FEATURE_NAMES)].to_numpy(dtype=float),
        rtol=1e-6,
        atol=1e-6,
        equal_nan=True,
    )
    episode = build_episodes(
        [
            {
                "subject_id": "FICTIONAL",
                "event_id": "FICTIONAL_EVENT",
                "timestamp": "2099-01-01T15:00:00Z",
                "outcome_code": 3,
            }
        ]
    )
    canonical, _ = generate_patient_manifest(reference.to_dict("records"), episode, cov, task)
    indexed = reference.set_index(pd.DatetimeIndex(pd.to_datetime(reference.timestamp, utc=True)))
    compact, flow, _ = window_index(indexed, episode, cov, task)
    assert len(canonical) == len(compact["end_rows"]) == flow["eligible_prediction_times"]
    np.testing.assert_array_equal([r["target"] for r in canonical], compact["target"])
    no_events, _, _ = window_index(indexed, [], cov, task)
    np.testing.assert_array_equal(compact["end_rows"], no_events["end_rows"])
    sequence = sequence_for_sample(reference.to_dict("records"), canonical[0], task)
    assert np.asarray(sequence).shape == (96, len(FEATURE_NAMES))


def test_resume_integrity_and_split_isolation(tmp_path):
    raw, output = tmp_path / "raw", tmp_path / "run"
    generate_raw(raw, subjects=2, days=2)
    first = run_pipeline(raw, output)
    assert run_pipeline(raw, output, resume=True) == first
    patients = json.loads((output / "patients.json").read_text())
    for patient in patients:
        assert patient["split"] == patient_split(patient["subject_id"], load_task())
    batches = list(sequence_batches(output, patients[0]["split"], batch_size=8))
    assert batches and batches[0]["X"].shape[1:] == (96, len(FEATURE_NAMES))
    with pytest.raises(ValueError):
        list(sequence_batches(output, "test", training_only=True))
    with (raw / "telemetry.csv").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="changed"):
        run_pipeline(raw, output, resume=True)
    (output / "quality.json").write_text("{}")
    with pytest.raises(ValueError, match="integrity"):
        verify(output)


def test_complete_raw_to_model_cli(tmp_path):
    output = tmp_path / "full"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ai_cvd.pipeline",
            "demo",
            "--subjects",
            "6",
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "raw-to-evaluation pipeline complete" in result.stdout
    record = verify(output / "processed")
    assert record["subjects"] == 6 and record["raw_rows"] > 10000
    quality = json.loads((output / "processed/quality.json").read_text())
    assert (
        quality["measurement"]["late_arrivals"] > 0 and quality["measurement"]["device_flagged"] > 0
    )
    evaluation = json.loads((output / "evaluation/metrics.json").read_text())
    assert evaluation["episode_count"] == 12
    assert not evaluation["manuscript_result_reproduction"]
    assert set(evaluation["evaluation"]) == {"tabular", "grud", "mtan"}


def test_interrupted_run_resumes_without_duplicate_ingestion(tmp_path, monkeypatch):
    from ai_cvd.pipeline import runner

    raw, output = tmp_path / "raw", tmp_path / "run"
    expected = generate_raw(raw, subjects=1, days=1)
    original = runner.write_parquet

    def interrupted(*args, **kwargs):
        raise RuntimeError("simulated storage interruption")

    monkeypatch.setattr(runner, "write_parquet", interrupted)
    with pytest.raises(RuntimeError, match="simulated"):
        run_pipeline(raw, output)
    assert json.loads((output / "run.json").read_text())["status"] == "incomplete"
    assert not (output / ".run.lock").exists()
    monkeypatch.setattr(runner, "write_parquet", original)
    (output / ".run.lock").write_text("another writer")
    with pytest.raises(FileExistsError):
        run_pipeline(raw, output, resume=True)
    assert (output / ".run.lock").read_text() == "another writer"
    (output / ".run.lock").unlink()
    result = run_pipeline(raw, output, resume=True)
    assert result["raw_rows"] == expected["raw_rows"]
    with connect(output / "telemetry.duckdb", read_only=True) as db:
        assert db.execute("SELECT count(*) FROM ingested_files").fetchone()[0] == 1
