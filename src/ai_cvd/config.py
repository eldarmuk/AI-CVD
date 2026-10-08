"""Load the packaged fictional-demo configuration."""

import tomllib
from importlib.resources import files

from ai_cvd.features.episode import CHANNELS


def public_configuration():
    config = tomllib.loads(files("ai_cvd").joinpath("synthetic.toml").read_text(encoding="utf-8"))
    if config["public_synthetic"] != {
        "synthetic_only": True,
        "fixture_year": 2099,
        "not_manuscript_result_reproduction": True,
    }:
        raise ValueError("Synthetic configuration identity differs")
    spec = config["features"]
    if tuple(spec["channel_order"]) != CHANNELS:
        raise ValueError("Channel order differs")
    names = (
        [
            f"{channel}__{suffix}"
            for group in (
                "absolute_suffix_order",
                "relative_suffix_order",
                "observation_suffix_order",
            )
            for channel in CHANNELS
            for suffix in spec[group]
        ]
        + spec["steps_names"]
        + spec["clock_names"]
    )
    log_suffixes = tuple("__" + suffix for suffix in spec["log1p_suffixes"])
    config["features"]["fields"] = [
        {
            "name": name,
            "transform": "log1p"
            if name.endswith(log_suffixes) or name in spec["log1p_names"]
            else "identity",
        }
        for name in names
    ]
    context = [f"{c}__delta_observation_rate_H" for c in CHANNELS]
    context += [f"{c}__{s}" for c in CHANNELS for s in spec["observation_suffix_order"]]
    context += spec["steps_names"] + spec["clock_names"]
    flags = [f"{c}__delta_observation_rate_H__missing" for c in CHANNELS]
    flags += ["steps__sum_H__missing", "steps__max_delta_interval_H__missing"]
    subset = config["feature_subsets"]["acquisition_activity_baseline_quality_context"]
    subset.update(raw_names=context, missing_indicator_names=flags)
    if len(names) != 90 or len(context) + len(flags) != 56:
        raise ValueError("Feature dimensions differ")
    return config
