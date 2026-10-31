import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hr_change.baselines import predict
from hr_change.contracts import Prediction
from hr_change.evaluation import classify_delta, evaluate, reference_label
from hr_change.synthetic import synthetic_episodes


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.episodes = synthetic_episodes()
        self.predictions = [
            predict(episode.model_input(), "last_observation") for episode in self.episodes
        ]

    def test_boundary_rules_and_reference_ambiguity(self):
        self.assertEqual(classify_delta(10, 10), "increase")
        self.assertEqual(classify_delta(-10, 10), "decrease")
        self.assertEqual(classify_delta(9.99, 10), "stable")
        self.assertEqual(reference_label(self.episodes[5], 10), "indeterminate")
        self.assertEqual(reference_label(self.episodes[6], 10), "unavailable")
        for threshold in (0, -1, float("nan"), True):
            with self.assertRaises(ValueError):
                classify_delta(1, threshold)

    def test_known_denominators_preserve_abstained_changes(self):
        result = evaluate(self.episodes, self.predictions, 10)
        counts = result["counts"]
        self.assertEqual(counts["episodes"], 10)
        self.assertEqual(counts["reference_eligible"], 8)
        self.assertEqual(counts["reference_indeterminate"], 1)
        self.assertEqual(counts["reference_unavailable"], 1)
        self.assertEqual(counts["true_changes"], 5)
        self.assertEqual(counts["detected_changes"], 3)
        self.assertEqual(counts["missed_changes"], 2)
        self.assertEqual(counts["changes_abstained"], 1)
        self.assertEqual(counts["wrong_direction"], 1)
        self.assertEqual(counts["false_changes"], 1)
        self.assertEqual(result["rates"]["change_sensitivity_all_reference_changes"], 3 / 5)

    def test_abstaining_everywhere_cannot_erase_true_changes(self):
        predictions = [
            Prediction(e.episode_id, None, None, "no_supported_decision") for e in self.episodes
        ]
        result = evaluate(self.episodes, predictions, 10)
        self.assertEqual(result["counts"]["true_changes"], 5)
        self.assertEqual(result["counts"]["changes_abstained"], 5)
        self.assertEqual(result["counts"]["missed_changes"], 5)
        self.assertEqual(result["rates"]["change_sensitivity_all_reference_changes"], 0)
        self.assertEqual(result["rates"]["coverage_reference_eligible"], 0)
        self.assertIsNone(result["rates"]["wrong_change_fraction_of_accepted_change_calls"])

    def test_rejecting_wrong_prediction_changes_coverage_not_denominator(self):
        predictions = self.predictions.copy()
        predictions[7] = replace(predictions[7], abstain_reason="illustration")
        result = evaluate(self.episodes, predictions, 10)
        self.assertEqual(result["counts"]["true_changes"], 5)
        self.assertEqual(result["counts"]["missed_changes"], 2)
        self.assertEqual(result["counts"]["wrong_direction"], 0)
        self.assertEqual(result["counts"]["changes_abstained"], 2)
        self.assertEqual(result["numerical_errors"]["all_paired_estimate_episodes"], 8)
        self.assertEqual(result["numerical_errors"]["accepted_paired_estimate_episodes"], 7)

    def test_fixed_bias_point_error_is_not_change_error(self):
        result = evaluate([self.episodes[8]], [self.predictions[8]], 10)
        self.assertEqual(result["numerical_errors"]["all_paired_point_mae_bpm"], 20)
        self.assertEqual(result["numerical_errors"]["all_change_mae_bpm"], 0)

    def test_ref_indeterminate_not_silently_treated_as_stable(self):
        result = evaluate([self.episodes[5]], [self.predictions[5]], 10)
        self.assertEqual(result["counts"]["reference_stable"], 0)
        self.assertEqual(result["counts"]["reference_indeterminate"], 1)
        self.assertIsNone(result["rates"]["coverage_reference_eligible"])

    def test_missing_or_duplicated_predictions_rejected(self):
        with self.assertRaisesRegex(ValueError, "every episode"):
            evaluate(self.episodes, self.predictions[:-1], 10)
        with self.assertRaisesRegex(ValueError, "unique"):
            evaluate(self.episodes, self.predictions + [self.predictions[0]], 10)

    def test_synthetic_same_measurements_need_independent_reference(self):
        # These two deliberately invented cases illustrate why values alone
        # cannot identify whether correction is right; not empirical evidence.
        self.assertEqual(self.predictions[1].recent_bpm, self.predictions[3].recent_bpm)
        self.assertEqual(reference_label(self.episodes[1], 10), "increase")
        self.assertEqual(reference_label(self.episodes[3], 10), "stable")


if __name__ == "__main__":
    unittest.main()
