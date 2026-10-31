"""Small explicit data contracts for causal inputs and offline reference evidence."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence


def finite_number(value: float, name: str, *, positive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else ""))


def nonnegative_time(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer recording-relative second")


@dataclass(frozen=True)
class Window:
    """Half-open recording-relative interval [start_s, end_s)."""

    start_s: int
    end_s: int

    def validate(self) -> None:
        nonnegative_time(self.start_s, "window start")
        nonnegative_time(self.end_s, "window end")
        if self.start_s >= self.end_s:
            raise ValueError("window start must precede its end")


@dataclass(frozen=True)
class Observation:
    timestamp_s: int
    available_at_s: int
    value_bpm: float

    def validate(self, window: Window, decision_time_s: int) -> None:
        nonnegative_time(self.timestamp_s, "observation time")
        nonnegative_time(self.available_at_s, "availability time")
        finite_number(self.value_bpm, "observed HR", positive=True)
        if not window.start_s <= self.timestamp_s < window.end_s:
            raise ValueError("observation lies outside its half-open input window")
        if self.available_at_s < self.timestamp_s:
            raise ValueError("availability cannot precede the observation")
        if self.available_at_s > decision_time_s:
            raise ValueError("future or delayed input unavailable at decision time")


@dataclass(frozen=True)
class ReferenceSummary:
    """Offline reference summary and explicit uncertainty bound.

    A bound is a supplied assumption; this class does not establish its coverage.
    Adapters must document ECG quality, synchronization and summary construction.
    None represents an unavailable reference, never an imputed negative label.
    """

    value_bpm: float | None
    bound_bpm: float | None
    n_valid: int

    def validate(self) -> None:
        if isinstance(self.n_valid, bool) or not isinstance(self.n_valid, int):
            raise ValueError("reference n_valid must be an integer")
        if self.value_bpm is None:
            if self.bound_bpm is not None or self.n_valid != 0:
                raise ValueError("unavailable reference requires no bound and zero valid samples")
            return
        finite_number(self.value_bpm, "reference HR", positive=True)
        finite_number(self.bound_bpm, "reference bound")
        if self.bound_bpm < 0 or self.n_valid < 1:
            raise ValueError("reference bound must be nonnegative and n_valid positive")


@dataclass(frozen=True)
class CausalInput:
    """Reference-free model view; participant_key is canonical across sources."""

    episode_id: str
    participant_key: str
    dataset_id: str
    decision_time_s: int
    baseline_window: Window
    recent_window: Window
    baseline_observations: tuple[Observation, ...]
    recent_observations: tuple[Observation, ...]

    def validate(self) -> None:
        for name in ("episode_id", "participant_key", "dataset_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        nonnegative_time(self.decision_time_s, "decision time")
        self.baseline_window.validate()
        self.recent_window.validate()
        if self.baseline_window.end_s > self.recent_window.start_s:
            raise ValueError("baseline and recent windows overlap or are out of order")
        if self.recent_window.end_s > self.decision_time_s:
            raise ValueError("input window extends beyond decision time")
        for window, observations in (
            (self.baseline_window, self.baseline_observations),
            (self.recent_window, self.recent_observations),
        ):
            seen = set()
            previous = -1
            for observation in observations:
                observation.validate(window, self.decision_time_s)
                if observation.timestamp_s in seen:
                    raise ValueError(
                        "duplicate observation timestamp; resolve upstream with provenance"
                    )
                if observation.timestamp_s < previous:
                    raise ValueError("observations must be ordered by measurement time")
                seen.add(observation.timestamp_s)
                previous = observation.timestamp_s


@dataclass(frozen=True)
class PairedEpisode(CausalInput):
    """One comparison plus offline reference evidence, reserved for evaluation."""

    baseline_reference: ReferenceSummary
    recent_reference: ReferenceSummary

    def validate(self) -> None:
        super().validate()
        self.baseline_reference.validate()
        self.recent_reference.validate()

    def model_input(self) -> CausalInput:
        self.validate()
        return CausalInput(
            self.episode_id,
            self.participant_key,
            self.dataset_id,
            self.decision_time_s,
            self.baseline_window,
            self.recent_window,
            self.baseline_observations,
            self.recent_observations,
        )

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["baseline_observations"] = list(payload["baseline_observations"])
        payload["recent_observations"] = list(payload["recent_observations"])
        return payload

    @classmethod
    def from_dict(cls, item: dict) -> PairedEpisode:
        episode = cls(
            episode_id=item["episode_id"],
            participant_key=item["participant_key"],
            dataset_id=item["dataset_id"],
            decision_time_s=item["decision_time_s"],
            baseline_window=Window(**item["baseline_window"]),
            recent_window=Window(**item["recent_window"]),
            baseline_observations=tuple(Observation(**x) for x in item["baseline_observations"]),
            recent_observations=tuple(Observation(**x) for x in item["recent_observations"]),
            baseline_reference=ReferenceSummary(**item["baseline_reference"]),
            recent_reference=ReferenceSummary(**item["recent_reference"]),
        )
        episode.validate()
        return episode


@dataclass(frozen=True)
class Prediction:
    episode_id: str
    baseline_bpm: float | None
    recent_bpm: float | None
    abstain_reason: str | None = None

    @property
    def accepted(self) -> bool:
        return self.abstain_reason is None

    def validate(self) -> None:
        if not isinstance(self.episode_id, str) or not self.episode_id.strip():
            raise ValueError("prediction requires an episode identifier")
        if (self.baseline_bpm is None) != (self.recent_bpm is None):
            raise ValueError("paired estimates must both be present or both absent")
        if self.baseline_bpm is None:
            if self.accepted:
                raise ValueError("accepted prediction must contain paired estimates")
        else:
            finite_number(self.baseline_bpm, "predicted baseline HR", positive=True)
            finite_number(self.recent_bpm, "predicted recent HR", positive=True)
        if not self.accepted and (
            not isinstance(self.abstain_reason, str) or not self.abstain_reason.strip()
        ):
            raise ValueError("abstention requires a nonempty reason")


def validate_participant_splits(splits: Mapping[str, Sequence[PairedEpisode]]) -> None:
    """Reject any participant or episode crossing partitions, even across datasets."""
    participants: dict[str, str] = {}
    episodes: set[str] = set()
    for split_name, partition in splits.items():
        if not isinstance(split_name, str) or not split_name.strip():
            raise ValueError("partition names must be nonempty")
        for episode in partition:
            episode.validate()
            if episode.episode_id in episodes:
                raise ValueError("episode occurs more than once across partitions")
            episodes.add(episode.episode_id)
            prior = participants.setdefault(episode.participant_key, split_name)
            if prior != split_name:
                raise ValueError("participant overlap across partitions")
