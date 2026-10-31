"""Evaluate all reference-eligible comparisons, preserving abstention denominators."""

from __future__ import annotations

from statistics import mean
from typing import Sequence

from .contracts import PairedEpisode, Prediction, finite_number


def classify_delta(delta: float, threshold_bpm: float) -> str:
    finite_number(delta, "HR change")
    finite_number(threshold_bpm, "change threshold", positive=True)
    if delta >= threshold_bpm:
        return "increase"
    if delta <= -threshold_bpm:
        return "decrease"
    return "stable"


def reference_label(episode: PairedEpisode, threshold_bpm: float) -> str:
    episode.validate()
    finite_number(threshold_bpm, "change threshold", positive=True)
    baseline, recent = episode.baseline_reference, episode.recent_reference
    if baseline.value_bpm is None or recent.value_bpm is None:
        return "unavailable"
    delta = recent.value_bpm - baseline.value_bpm
    # Sum of the supplied bounds is conservative for their difference. This is
    # not a learned confidence interval and asserts no empirical coverage.
    bound = baseline.bound_bpm + recent.bound_bpm
    lower, upper = delta - bound, delta + bound
    if lower >= threshold_bpm:
        return "increase"
    if upper <= -threshold_bpm:
        return "decrease"
    if lower > -threshold_bpm and upper < threshold_bpm:
        return "stable"
    return "indeterminate"


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def evaluate(
    episodes: Sequence[PairedEpisode], predictions: Sequence[Prediction], threshold_bpm: float
) -> dict:
    finite_number(threshold_bpm, "change threshold", positive=True)
    episode_ids = [episode.episode_id for episode in episodes]
    prediction_ids = [prediction.episode_id for prediction in predictions]
    if len(set(episode_ids)) != len(episode_ids) or len(set(prediction_ids)) != len(prediction_ids):
        raise ValueError("episode and prediction identifiers must be unique")
    if set(episode_ids) != set(prediction_ids):
        raise ValueError("every episode needs exactly one prediction, including abstentions")
    by_id = {prediction.episode_id: prediction for prediction in predictions}
    counts = dict.fromkeys(
        [
            "episodes",
            "reference_eligible",
            "reference_indeterminate",
            "reference_unavailable",
            "true_changes",
            "reference_stable",
            "accepted_eligible",
            "abstained_eligible",
            "detected_changes",
            "missed_changes",
            "changes_abstained",
            "changes_called_stable",
            "wrong_direction",
            "false_changes",
            "accepted_change_calls",
        ],
        0,
    )
    points: list[float] = []
    changes: list[float] = []
    accepted_points: list[float] = []
    accepted_changes: list[float] = []
    rows = []
    for episode in episodes:
        episode.validate()
        prediction = by_id[episode.episode_id]
        prediction.validate()
        label = reference_label(episode, threshold_bpm)
        counts["episodes"] += 1
        point_error = change_error = None
        predicted_delta = None
        decision = "abstain"
        if prediction.baseline_bpm is not None:
            predicted_delta = prediction.recent_bpm - prediction.baseline_bpm
            if prediction.accepted:
                decision = classify_delta(predicted_delta, threshold_bpm)
            baseline, recent = episode.baseline_reference, episode.recent_reference
            if baseline.value_bpm is not None and recent.value_bpm is not None:
                point_error = (
                    abs(prediction.baseline_bpm - baseline.value_bpm)
                    + abs(prediction.recent_bpm - recent.value_bpm)
                ) / 2
                change_error = abs(predicted_delta - (recent.value_bpm - baseline.value_bpm))
                points.append(point_error)
                changes.append(change_error)
                if prediction.accepted:
                    accepted_points.append(point_error)
                    accepted_changes.append(change_error)
        if label in ("unavailable", "indeterminate"):
            counts[f"reference_{label}"] += 1
        else:
            counts["reference_eligible"] += 1
            counts["accepted_eligible" if prediction.accepted else "abstained_eligible"] += 1
            is_change = label != "stable"
            counts["true_changes" if is_change else "reference_stable"] += 1
            if decision in ("increase", "decrease"):
                counts["accepted_change_calls"] += 1
            if is_change:
                if decision == label:
                    counts["detected_changes"] += 1
                else:
                    counts["missed_changes"] += 1
                    reason = (
                        "changes_abstained"
                        if decision == "abstain"
                        else "changes_called_stable"
                        if decision == "stable"
                        else "wrong_direction"
                    )
                    counts[reason] += 1
            elif decision in ("increase", "decrease"):
                counts["false_changes"] += 1
        rows.append(
            {
                "episode_id": episode.episode_id,
                "reference_label": label,
                "decision": decision,
                "abstain_reason": prediction.abstain_reason,
                "predicted_delta_bpm": predicted_delta,
                "paired_point_mae_bpm": point_error,
                "change_absolute_error_bpm": change_error,
            }
        )
    return {
        "threshold_bpm": threshold_bpm,
        "counts": counts,
        "rates": {
            "coverage_reference_eligible": _ratio(
                counts["accepted_eligible"], counts["reference_eligible"]
            ),
            "change_sensitivity_all_reference_changes": _ratio(
                counts["detected_changes"], counts["true_changes"]
            ),
            "changes_lost_to_abstention_fraction": _ratio(
                counts["changes_abstained"], counts["true_changes"]
            ),
            "false_change_rate_reference_stable": _ratio(
                counts["false_changes"], counts["reference_stable"]
            ),
            "wrong_change_fraction_of_accepted_change_calls": _ratio(
                counts["false_changes"] + counts["wrong_direction"], counts["accepted_change_calls"]
            ),
        },
        "numerical_errors": {
            "all_paired_estimate_episodes": len(points),
            "all_paired_point_mae_bpm": mean(points) if points else None,
            "all_change_mae_bpm": mean(changes) if changes else None,
            "accepted_paired_estimate_episodes": len(accepted_points),
            "accepted_paired_point_mae_bpm": mean(accepted_points) if accepted_points else None,
            "accepted_change_mae_bpm": mean(accepted_changes) if accepted_changes else None,
        },
        "episodes": rows,
    }
