"""Fictional raw telemetry exports, including known engineering faults."""

import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from ai_cvd.data.episodes import SYNTHETIC_PREFIX

from .episodes import iso
from .storage import RAW_COLUMNS, UNITS


def generate_raw(destination, subjects=12, days=10, seed=17, immediate=False):
    destination = Path(destination)
    if subjects < 1 or days < 1 or seed < 0:
        raise ValueError("Positive subjects/days and nonnegative seed required")
    destination.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    origin = datetime(2099, 1, 1, tzinfo=timezone.utc)
    coverage, alerts = [], []
    count = 0
    with (destination / "telemetry.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RAW_COLUMNS)
        writer.writeheader()
        for patient in range(subjects):
            sid = f"{SYNTHETIC_PREFIX}{patient:03d}"
            start = origin + timedelta(hours=patient)
            end = start + timedelta(days=days)
            coverage.append(
                {
                    "subject_id": sid,
                    "observation_start": iso(start),
                    "observation_end": iso(end),
                    "outcome_start": iso(start),
                    "outcome_end": iso(end),
                }
            )
            for visit in range(2):
                anchor = start + timedelta(days=min(8 + visit, days - 1), hours=12)
                if anchor < end:
                    for member in range(2):
                        alerts.append(
                            {
                                "subject_id": sid,
                                "event_id": f"SYNTHETIC_EVENT_{patient}_{visit}_{member}",
                                "timestamp": iso(anchor + timedelta(minutes=5 * member)),
                                "outcome_code": 3 if member and (patient + visit) % 2 else 1,
                            }
                        )
            for tick in range(days * 24 * 6):
                at = start + timedelta(minutes=10 * tick + 1)
                for index, kind in enumerate(UNITS):
                    if (tick + index + patient) % 19 == 0:
                        continue  # Missing transmission, not an observed zero.
                    value = {
                        "Heartrate": 65 + (tick + patient) % 12 + rng.normal(0, 0.1),
                        "Temperature": 36 + (tick % 5) / 10,
                        "BloodPressure": "",
                        "Saturation": 95 + tick % 4,
                        "Steps": (tick % (24 * 6)) * 8,
                    }[kind]
                    row = dict(
                        zip(
                            RAW_COLUMNS,
                            [
                                f"SYNTHETIC_RECORD_{patient}_{tick}_{index}",
                                sid,
                                iso(at),
                                iso(at if immediate else at + timedelta(seconds=5)),
                                kind,
                                value,
                                115 + tick % 10 if kind == "BloodPressure" else "",
                                70 + tick % 8 if kind == "BloodPressure" else "",
                                UNITS[kind],
                                "ok",
                            ],
                            strict=True,
                        )
                    )
                    if tick % 101 == 0:
                        row["device_status"] = "off_body"
                    elif tick % 97 == 0 and kind == "Heartrate":
                        row["value"] = 9999
                    elif tick % 89 == 0 and kind == "BloodPressure":
                        row["dbp"] = 140  # Inverted pair is invalid, not clipped.
                    elif tick % 83 == 0 and not immediate:
                        row["available_at"] = iso(at + timedelta(minutes=8))
                    writer.writerow(row)
                    count += 1
                    if tick % 79 == 0:
                        writer.writerow(dict(row, record_id=row["record_id"] + "_RETRANSMISSION"))
                        count += 1
    for name, rows in [("coverage.csv", coverage), ("events.csv", alerts)]:
        with (destination / name).open("w", newline="", encoding="utf-8") as stream:
            columns = (
                list(rows[0]) if rows else ["subject_id", "event_id", "timestamp", "outcome_code"]
            )
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    source = {
        "synthetic": True,
        "fixture": "WHOLLY FICTIONAL; NOT CLINICAL DATA",
        "seed": seed,
        "contract": {
            "steps_semantics": "cumulative_counter",
            "counter_reset_policy": "daily",
            "source_timezone": "UTC",
            "steps_evidence": "fictional-generator-v1",
            "measurement_availability": "timestamp_is_available_at"
            if immediate
            else "explicit_available_at",
            "availability_evidence": "fictional-generator-v1",
            "coverage_evidence": "fictional-generator-v1",
            "clinical_history_policy": "exclude",
            "alert_time_semantics": "alarm_initiation_with_retrospective_classification",
            "alert_time_evidence": "fictional-generator-v1",
        },
    }
    (destination / "source.json").write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")
    return {"subjects": subjects, "raw_rows": count, "events": len(alerts)}
