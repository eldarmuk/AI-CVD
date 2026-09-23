import copy
from datetime import timedelta
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from src.ai_cvd.task import load_task, Task
from src.ai_cvd.episodes import as_time, iso, build_episodes, event_time, classify_severity, write_episode_tables
from src.ai_cvd.features import build_features, FEATURE_NAMES, clean_measurement, validate_source_contract
from src.ai_cvd.dataset import (generate_patient_manifest, training_manifest, development_manifest, sample_id,
                                sequence_for_sample, patient_split, assert_patient_isolation, align_by_sample_id)
from src.ai_cvd.synthetic import fixture
from src.ai_cvd.cli import build_run, verify_run, read_jsonl, SQLiteSource, localize


class ScientificPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.task = load_task()
        cls.source, cls.coverages, cls.histories, cls.contract = fixture(cls.task)
        cls.coverage = cls.coverages[0]
        cls.sid = cls.coverage["senior_id"]
        cls.start = as_time(cls.coverage["enrollment_time"])
        cls.rows = list(build_features(cls.source.measurements(cls.sid), cls.coverage, [cls.histories[0]], cls.task, cls.contract))
        cls.episodes = build_episodes(cls.source.alerts(cls.sid), cls.task)
        cls.stream, cls.flow = generate_patient_manifest(cls.rows, cls.episodes, cls.coverage, cls.task)

    def test_task_duration_assertion(self):
        bad = dict(self.task.values, lookback_minutes=1440)
        with self.assertRaisesRegex(ValueError, "sequence_steps"):
            Task(bad).validate()
        self.assertEqual(self.task.lookback_minutes, 480)

    def test_endpoint_cannot_silently_change(self):
        with self.assertRaises(ValueError):
            Task(dict(self.task.values, target_severities=[2, 3])).validate()
        self.assertNotEqual(self.task.identifier, Task(dict(self.task.values, version="test")).identifier)

    def alerts(self, severities, minutes):
        return [{"senior_id": "fiction", "alert_id": str(i), "alert_date": iso(self.start + timedelta(minutes=m)), "severity": s} for i, (s, m) in enumerate(zip(severities, minutes))]

    def test_level1_escalation_is_retained(self):
        e = build_episodes(self.alerts([1, 3], [0, 5]), self.task)[0]
        self.assertEqual((e["first_severity"], e["maximum_severity"], e["constituent_count"]), (1, 3, 2))
        self.assertEqual(event_time(e, [3]), self.start + timedelta(minutes=5))
        self.assertEqual(as_time(e["escalation_recorded_at"]), self.start + timedelta(minutes=5))

    def test_level2_escalation_and_later_downgrade(self):
        e = build_episodes(self.alerts([2, 3, 1], [0, 5, 10]), self.task)[0]
        self.assertEqual(e["maximum_severity"], 3)
        self.assertEqual(e["final_recorded_severity"], 1)
        self.assertEqual(event_time(e, [2, 3]), self.start)

    def test_accidental_alerts(self):
        episodes = build_episodes(self.alerts([0, 0, 0], [0, 5, 10]), self.task)
        self.assertEqual(len(episodes), 1)
        self.assertIsNone(event_time(episodes[0], [3]))

    def test_inclusive_chained_burst_boundaries(self):
        for gap, expected in [(9, 1), (10, 1), (11, 2)]:
            self.assertEqual(len(build_episodes(self.alerts([1, 3], [0, gap]), self.task)), expected)
        self.assertEqual(len(build_episodes(self.alerts([1, 2, 3], [0, 10, 20]), self.task)), 1)

    def test_duplicate_alert_ids_rejected(self):
        a = self.alerts([1], [0])
        with self.assertRaises(ValueError):
            build_episodes(a+a, self.task)

    def test_severity_rules(self):
        self.assertEqual(classify_severity("Nie nawiązano kontaktu"), 2)
        self.assertEqual(classify_severity("Alarm przypadkowy; zdecydowano wezwać ZRM"), 3)
        self.assertNotEqual(classify_severity("brak wskazań do interwencji ZRM"), 3)
        self.assertEqual(classify_severity(None), -1)

    def test_episode_sql_preserves_members_and_refuses_overwrite(self):
        connection = sqlite3.connect(":memory:")
        write_episode_tables(connection, self.episodes)
        self.assertEqual(connection.execute("SELECT maximum_severity,constituent_count FROM alert_episodes_v2").fetchone(), (3, 2))
        self.assertEqual(connection.execute("SELECT count(*) FROM alert_episode_members_v2").fetchone()[0], 2)
        with self.assertRaises(sqlite3.OperationalError):
            write_episode_tables(connection, self.episodes)
        connection.close()

    def test_shock_index_is_hr_over_sbp(self):
        self.assertAlmostEqual(self.rows[0]["shock_index"], 70/120)

    def test_recency_and_bucket_availability(self):
        first, invalid = self.rows[:2]
        self.assertEqual(first["time_since_last_heartrate"], 2)
        self.assertEqual(invalid["time_since_last_heartrate"], 7)
        self.assertEqual(invalid["observed_heartrate"], 0)
        self.assertEqual(as_time(first["available_at"]) - as_time(first["timestamp"]), timedelta(minutes=5))
        for row in self.rows:
            self.assertTrue(all(row[k] is None or row[k] >= 0 for k in row if k.startswith("time_since_last_")))

    def test_missing_steps_and_observed_zero(self):
        self.assertEqual((self.rows[0]["steps"], self.rows[0]["observed_steps"]), (0, 1))
        self.assertEqual((self.rows[1]["steps"], self.rows[1]["observed_steps"]), (None, 0))

    def test_spo2_both_bounds(self):
        for value in [49, 101, float("inf")]:
            self.assertIsNone(clean_measurement({"type": "Saturation", "value": value}, self.task)["saturation"])
        for value in [50, 100]:
            self.assertEqual(clean_measurement({"type": "Saturation", "value": value}, self.task)["saturation"], value)

    def test_pp_rule_and_invalid_bp_do_not_reset(self):
        self.assertEqual(self.rows[0]["pulse_pressure"], 40)
        self.assertIsNone(self.rows[1]["pulse_pressure"])
        self.assertIsNone(self.rows[2]["pulse_pressure"])
        self.assertEqual(self.rows[2]["time_since_last_pulse_pressure"], 12)
        self.assertEqual(self.rows[1]["time_since_last_sbp"], 7)

    def test_duplicate_measurements_not_double_counted(self):
        row = {"senior_id": self.sid, "date": iso(self.start + timedelta(minutes=3)), "type": "Steps", "value": 12}
        rows = list(build_features([row, dict(row)], self.coverage, [], self.task, self.contract))
        self.assertEqual(rows[0]["steps"], 12)

    def test_unverified_steps_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "Unverified Steps"):
            validate_source_contract(dict(self.contract, steps_semantics="unknown"))

    def test_cumulative_counter_reset_is_unknown(self):
        data = [{"senior_id": self.sid, "date": iso(self.start + timedelta(minutes=m)), "type": "Steps", "value": v} for m, v in [(3, 100), (8, 110), (13, 5), (18, 9)]]
        rows = list(build_features(data, self.coverage, [], self.task, dict(self.contract, steps_semantics="cumulative_counter", counter_reset_policy="daily")))
        self.assertEqual([r["steps"] for r in rows[:4]], [None, 10, None, 4])

    def test_delayed_measurement_never_backfilled(self):
        row = {"senior_id": self.sid, "date": iso(self.start + timedelta(minutes=3)), "available_at": iso(self.start + timedelta(minutes=8)), "type": "Heartrate", "value": 70}
        rows = list(build_features([row], self.coverage, [], self.task, dict(self.contract, measurement_availability="explicit_available_at")))
        self.assertTrue(all(r["heartrate"] is None for r in rows))

    def test_future_clinical_history_not_backfilled(self):
        h = {"senior_id": self.sid, "available_at": iso(self.start + timedelta(hours=30)), "values": {"cardiovascular": 1}}
        rows = list(build_features([], self.coverage, [h], self.task, self.contract))
        self.assertIsNone(rows[359]["cardiovascular"])
        self.assertEqual(rows[360]["cardiovascular"], 1)
        with self.assertRaises(ValueError):
            list(build_features([], self.coverage, [h], self.task, dict(self.contract, clinical_history_policy="verified_baseline", baseline_evidence="test")))

    def test_exactly_48_positive_prediction_times(self):
        event = self.start + timedelta(hours=48)
        positives = [as_time(r["prediction_time"]) for r in self.stream if r["target"]]
        expected = [event - timedelta(minutes=240) + timedelta(minutes=5*i) for i in range(48)]
        self.assertEqual(positives, expected)  # Historical sampler returns only one.
        self.assertEqual(len({r["episode_id"] for r in self.stream if r["target"]}), 1)

    def test_horizon_boundary_and_event_at_t(self):
        lookup = {as_time(r["prediction_time"]): r["target"] for r in self.stream}
        event = self.start + timedelta(hours=48)
        self.assertEqual(lookup[event - timedelta(minutes=245)], 0)
        self.assertEqual(lookup[event - timedelta(minutes=240)], 1)
        self.assertEqual(lookup[event], 0)

    def test_candidate_times_independent_of_labels(self):
        no_alerts, _ = generate_patient_manifest(self.rows, [], self.coverage, self.task)
        self.assertEqual([r["sample_id"] for r in no_alerts], [r["sample_id"] for r in self.stream])
        self.assertFalse(any(r["target"] for r in no_alerts))

    def test_future_density_does_not_change_past_eligibility(self):
        altered = copy.deepcopy(self.rows)
        for row in altered:
            if as_time(row["timestamp"]) >= self.start + timedelta(hours=40):
                row["observed_heartrate"] = row["observed_sbp"] = row["observed_dbp"] = 0
        other, _ = generate_patient_manifest(altered, self.episodes, self.coverage, self.task)
        self.assertEqual([r["sample_id"] for r in other], [r["sample_id"] for r in self.stream])

    def test_insufficient_followup_is_censored(self):
        end = as_time(self.coverage["outcome_coverage_end"])
        self.assertTrue(all(as_time(r["prediction_time"]) + timedelta(minutes=240) <= end for r in self.stream))
        self.assertEqual(self.flow["excluded_insufficient_follow_up"], 48)

    def test_input_interval_masks_feature_order_and_ids(self):
        sample = self.stream[0]
        sequence = sequence_for_sample(self.rows, sample, self.task)
        self.assertEqual(len(sequence), 96)
        self.assertEqual(len(sequence[0]), len(FEATURE_NAMES))
        self.assertFalse(any("label" in f or f in {"target", "senior_id", "episode_id"} for f in FEATURE_NAMES))
        self.assertEqual(len({r["sample_id"] for r in self.stream}), len(self.stream))
        self.assertEqual(sample_id(self.sid, as_time(sample["prediction_time"])), sample["sample_id"])
        start = as_time(sample["input_start"])
        input_rows = [r for r in self.rows if start <= as_time(r["timestamp"]) < as_time(sample["prediction_time"])]
        self.assertEqual(sequence[0], [input_rows[0][f] for f in FEATURE_NAMES])

    def test_patient_isolation_and_hash_split(self):
        rows = [{"senior_id": str(i), "split": patient_split(str(i), self.task)} for i in range(100)]
        assert_patient_isolation(rows)
        self.assertEqual(set(r["split"] for r in rows), {"train", "validation", "test"})
        with self.assertRaisesRegex(ValueError, "overlap"):
            assert_patient_isolation(rows + [dict(rows[0], split="test" if rows[0]["split"] != "test" else "train")])

    def test_id_join_repairs_permutation_and_rejects_ambiguity(self):
        rows = [{"sample_id": "a", "task_identifier": self.task.identifier, "endpoint": "level3", "source_manifest_sha256": "fixture", "score": 1}, {"sample_id": "b", "task_identifier": self.task.identifier, "endpoint": "level3", "source_manifest_sha256": "fixture", "score": 2}]
        self.assertEqual([r["score"] for r in align_by_sample_id(["a", "b"], rows[::-1], task_identifier=self.task.identifier, endpoint="level3", source_manifest_sha256="fixture")], [1, 2])
        for bad in [rows[:1], rows+rows[:1], [dict(rows[0], task_identifier="old"), rows[1]], [dict(rows[0], endpoint="level2_or_level3"), rows[1]], [dict(rows[0], source_manifest_sha256="other"), rows[1]]]:
            with self.assertRaises(ValueError):
                align_by_sample_id(["a", "b"], bad, task_identifier=self.task.identifier, endpoint="level3", source_manifest_sha256="fixture")

    def test_retrospective_training_policy_separate_from_stream(self):
        train_stream = [dict(r, split="train") for r in self.stream]
        normal = training_manifest(train_stream, self.episodes, self.task, coverage=self.coverage)
        self.assertGreater(len(normal), 0)
        self.assertLess(len(normal), len(train_stream))
        self.assertEqual(training_manifest(train_stream, self.episodes, self.task, "never_event_patients", self.coverage), [])
        self.assertFalse(any(r["target"] for r in normal))

    def test_development_manifest_never_samples_test(self):
        data = [dict(r, split="validation") for r in self.stream] + [dict(r, split="test") for r in self.stream]
        selected = development_manifest(data, per_class=10)
        self.assertEqual(len(selected), 20)
        self.assertEqual(sum(r["target"] for r in selected), 10)
        self.assertEqual({r["split"] for r in selected}, {"validation"})

    def test_new_run_roundtrip_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"run"
            build_run(self.source, self.coverages, self.histories, self.task, self.contract, output, source_snapshot_id="synthetic")
            meta = verify_run(output, self.task)
            self.assertEqual(tuple(meta["feature_names"]), FEATURE_NAMES)
            manifest = list(read_jsonl(output/"stream_manifest.jsonl"))
            assert_patient_isolation(manifest)
            self.assertEqual(len(manifest), len({r["sample_id"] for r in manifest}))
            flow = json.loads((output/"cohort_flow.json").read_text())["counts"]
            self.assertGreater(flow["events_lost_insufficient_history"], 0)
            self.assertGreater(flow["events_lost_insufficient_follow_up"], 0)
            with self.assertRaises(FileExistsError):
                build_run(self.source, self.coverages, self.histories, self.task, self.contract, output)
            with open(output/"stream_manifest.jsonl", "a") as handle:
                handle.write("{}\n")
            with self.assertRaisesRegex(ValueError, "checksum"):
                verify_run(output, self.task)


if __name__ == "__main__":
    unittest.main()
