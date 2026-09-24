"""The TOML task is authoritative; hashes bind every artifact to its contents."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import math
import tomllib

DEFAULT_TASK = Path(__file__).resolve().parents[2] / "configs/tasks/level3_4h.toml"


@dataclass(frozen=True)
class Task:
    values: dict

    def __getattr__(self, name):
        try:
            return object.__getattribute__(self, "values")[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.values, sort_keys=True).encode()).hexdigest()

    @property
    def identifier(self):
        return f"{self.task_id}@{self.version}:{self.fingerprint}"

    def validate(self):
        if self.grid_minutes * self.sequence_steps != self.lookback_minutes:
            raise ValueError("grid_minutes * sequence_steps must equal lookback_minutes")
        if (self.grid_minutes, self.sequence_steps, self.lookback_minutes, self.horizon_minutes) != (5, 96, 480, 240):
            raise ValueError("This primary implementation requires the frozen 5/96/480/240 task")
        if (self.hr_sd_minutes, self.bp_trend_minutes, self.steps_sum_minutes) != (240, 180, 360):
            raise ValueError("Rolling durations must match the named feature schema")
        for name in ("grid_minutes", "sequence_steps", "horizon_minutes", "prediction_stride_steps", "run_in_minutes"):
            if not isinstance(self.values[name], int) or self.values[name] <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.run_in_minutes % self.grid_minutes:
            raise ValueError("run-in must align to the grid")
        if self.run_in_bp_definition != "valid_paired_pulse_pressure_bucket":
            raise ValueError("Unknown run-in BP observation definition")
        if self.endpoint != "level3" or self.target_severities != [3]:
            raise ValueError("Primary task must remain Level 3; request secondary endpoint explicitly")
        if self.secondary_severities != [2, 3] or self.secondary_endpoint != "level2_or_level3":
            raise ValueError("Unsupported secondary endpoint")
        if self.input_interval != "[t-lookback,t)" or self.target_interval != "(t,t+horizon]":
            raise ValueError("Unsupported interval semantics")
        if self.event_time != "first_alarm_with_retrospective_qualifying_classification" or self.burst_boundary != "inclusive_chained":
            raise ValueError("Unsupported episode semantics")
        if self.split_unit != "senior_id" or len(self.split_ratios) != 3:
            raise ValueError("Patient-level train/validation/test split required")
        if not math.isclose(sum(self.split_ratios), 1) or any(x <= 0 for x in self.split_ratios):
            raise ValueError("Split ratios must be positive and sum to one")
        if self.training_policy not in {"event_free_windows", "never_event_patients"}:
            raise ValueError("Unknown retrospective training policy")
        for bounds in self.validity.values():
            if len(bounds) != 2 or bounds[0] >= bounds[1]:
                raise ValueError("Invalid physiological validity bounds")
        return self


def load_task(path=DEFAULT_TASK):
    with open(path, "rb") as handle:
        return Task(tomllib.load(handle)).validate()
