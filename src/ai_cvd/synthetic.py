"""Public, wholly fictional fixture. No parameters fitted to private patients."""
from datetime import datetime, timedelta, timezone
from .episodes import iso


class MemorySource:
    def __init__(self, patients, measurements, alerts):
        self.ids, self.rows, self.events = patients, measurements, alerts

    def patients(self):
        return list(self.ids)

    def measurements(self, sid):
        return (dict(r) for r in self.rows if str(r["senior_id"]) == str(sid))

    def alerts(self, sid):
        return (dict(r) for r in self.events if str(r["senior_id"]) == str(sid))


def fixture(task):
    start = datetime(2030, 1, 1, tzinfo=timezone.utc)
    ids = ["fictional_" + str(i) for i in range(6)]
    measurements, alerts, coverages, histories = [], [], [], []
    for i, sid in enumerate(ids):
        coverages.append({"senior_id": sid, "enrollment_time": iso(start), "measurement_coverage_end": iso(start + timedelta(hours=72)), "outcome_coverage_start": iso(start), "outcome_coverage_end": iso(start + timedelta(hours=60 if i == 5 else 72))})
        histories.append({"senior_id": sid, "available_at": iso(start), "values": {"age": 70 + i, "gender": i % 2, "cardiovascular": i % 2}})
        for hour in range(72):
            at = start + timedelta(hours=hour, minutes=3)
            measurements.append({"senior_id": sid, "date": iso(at), "type": "Heartrate", "value": 70 + hour % 9})
            if hour % 2 == 0:
                measurements.append({"senior_id": sid, "date": iso(at), "type": "BloodPressure", "sbp": 120, "dbp": 80})
            if hour % 3 == 0:
                measurements.append({"senior_id": sid, "date": iso(at), "type": "Steps", "value": 0 if hour % 6 == 0 else 20})
        # Exact duplicate, invalid HR/SpO2, missing temperature, invalid/narrow BP.
        measurements.append(dict(measurements[-1]))
        for minute, kind, value in [(8, "Heartrate", 999), (8, "Saturation", 101), (13, "Saturation", 49), (18, "Saturation", 98)]:
            measurements.append({"senior_id": sid, "date": iso(start + timedelta(minutes=minute)), "type": kind, "value": value})
        measurements.extend([
            {"senior_id": sid, "date": iso(start + timedelta(minutes=8)), "type": "BloodPressure", "sbp": 80, "dbp": 90},
            {"senior_id": sid, "date": iso(start + timedelta(minutes=13)), "type": "BloodPressure", "sbp": 90, "dbp": 85},
        ])
        if i % 2 == 0:
            for suffix, hour, minute, severity in [("a", 47, 55, 1), ("b", 48, 0, 3)]:
                alerts.append({"senior_id": sid, "alert_id": f"{sid}_{suffix}", "alert_date": iso(start + timedelta(hours=hour, minutes=minute)), "severity": severity})
        if i == 1:
            alerts.extend({"senior_id": sid, "alert_id": f"accidental_{j}", "alert_date": iso(start + timedelta(hours=40, minutes=j*5)), "severity": 0} for j in range(3))
        if i == 3:
            alerts.append({"senior_id": sid, "alert_id": "early", "alert_date": iso(start + timedelta(hours=2)), "severity": 3})
        if i == 5:
            alerts.append({"senior_id": sid, "alert_id": "unobserved_followup", "alert_date": iso(start + timedelta(hours=70)), "severity": 3})
    contract = {"steps_semantics": "increments", "steps_evidence": "Fictional generator defines interval increments; not inferred from private source", "measurement_availability": "timestamp_is_available_at", "availability_evidence": "Synthetic immediate delivery", "coverage_evidence": "Synthetic observation intervals fixed before simulation", "clinical_history_policy": "dated_snapshots", "source_timezone": "UTC"}
    contract.update(alert_time_semantics="alert_timestamp_is_severity_recorded_at", alert_time_evidence="Fictional alert severity is recorded at creation")
    return MemorySource(ids, measurements, alerts), coverages, histories, contract
