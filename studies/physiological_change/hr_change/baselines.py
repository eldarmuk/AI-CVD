"""Two causal baselines without fitting, reference access or learned correction."""

from statistics import median

from .contracts import CausalInput, Prediction


def predict(episode: CausalInput, method: str, minimum_per_window: int = 1) -> Prediction:
    if type(episode) is not CausalInput:
        raise ValueError("predict requires a reference-free CausalInput, not evaluation data")
    episode.validate()
    if method not in ("last_observation", "window_median"):
        raise ValueError("unknown baseline")
    if (
        isinstance(minimum_per_window, bool)
        or not isinstance(minimum_per_window, int)
        or minimum_per_window < 1
    ):
        raise ValueError("minimum_per_window must be a positive integer")
    if (
        min(len(episode.baseline_observations), len(episode.recent_observations))
        < minimum_per_window
    ):
        return Prediction(episode.episode_id, None, None, "insufficient_observed_support")
    summaries = []
    for observations in (episode.baseline_observations, episode.recent_observations):
        values = [item.value_bpm for item in observations]
        summaries.append(values[-1] if method == "last_observation" else median(values))
    return Prediction(episode.episode_id, *summaries)
