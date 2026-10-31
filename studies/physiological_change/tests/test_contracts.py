import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hr_change.baselines import predict
from hr_change.contracts import (
    Observation,
    PairedEpisode,
    Prediction,
    ReferenceSummary,
    Window,
    validate_participant_splits,
)
from hr_change.synthetic import synthetic_episodes


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.episode = synthetic_episodes()[0]

    def test_future_window_rejected(self):
        with self.assertRaisesRegex(ValueError, "beyond decision"):
            replace(self.episode, decision_time_s=119).validate()

    def test_future_arrival_rejected_even_with_past_measurement(self):
        observation = Observation(80, 121, 75)
        with self.assertRaisesRegex(ValueError, "unavailable at decision"):
            replace(self.episode, recent_observations=(observation,)).validate()

    def test_overlapping_windows_rejected(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            replace(self.episode, baseline_window=Window(0, 70)).validate()

    def test_endpoint_outside_half_open_window(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            replace(self.episode, recent_observations=(Observation(120, 120, 75),)).validate()

    def test_duplicates_not_silently_averaged(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            replace(
                self.episode, recent_observations=(Observation(60, 60, 75), Observation(60, 60, 95))
            ).validate()

    def test_nonfinite_and_nonpositive_values_rejected(self):
        for value in (float("nan"), float("inf"), -1, 0, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                replace(self.episode, recent_observations=(Observation(60, 60, value),)).validate()

    def test_reference_absence_is_explicit(self):
        ReferenceSummary(None, None, 0).validate()
        with self.assertRaises(ValueError):
            ReferenceSummary(None, 2, 0).validate()
        with self.assertRaises(ValueError):
            ReferenceSummary(75, -1, 1).validate()

    def test_model_cannot_receive_reference_augmented_episode(self):
        with self.assertRaisesRegex(ValueError, "reference-free"):
            predict(self.episode, "last_observation")
        inputs = self.episode.model_input()
        self.assertFalse(hasattr(inputs, "baseline_reference"))
        self.assertEqual(predict(inputs, "last_observation").recent_bpm, 75)

    def test_reference_changes_do_not_change_prediction(self):
        changed = replace(self.episode, recent_reference=ReferenceSummary(200, 0, 10))
        self.assertEqual(
            predict(self.episode.model_input(), "window_median"),
            predict(changed.model_input(), "window_median"),
        )

    def test_participant_overlap_rejected_even_across_datasets(self):
        other = replace(self.episode, episode_id="new_id", dataset_id="another_dataset")
        with self.assertRaisesRegex(ValueError, "participant overlap"):
            validate_participant_splits({"development": [self.episode], "external": [other]})

    def test_valid_grouped_partitions(self):
        episodes = synthetic_episodes()
        validate_participant_splits({"development": episodes[:6], "external": episodes[6:]})

    def test_every_accepted_prediction_needs_estimates(self):
        with self.assertRaises(ValueError):
            Prediction("episode", None, None).validate()
        with self.assertRaises(ValueError):
            Prediction("episode", 75, None, "partial").validate()

    def test_serialization_roundtrip(self):
        self.assertEqual(PairedEpisode.from_dict(self.episode.to_dict()), self.episode)


if __name__ == "__main__":
    unittest.main()
