"""Run an end-to-end fictional telecare demonstration without private artifacts."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from ai_cvd.config import public_configuration
from ai_cvd.data.episodes import split_patients
from ai_cvd.data.synthetic import build_fixture
from ai_cvd.evaluation.metrics import binary_metrics, choose_threshold, operating_metrics
from ai_cvd.features.attention import MTANPreprocessor
from ai_cvd.features.temporal import TemporalPreprocessor
from ai_cvd.models.grud import make_model as make_grud
from ai_cvd.models.mtan import make_model as make_mtan
from ai_cvd.models.tabular import TabularBaseline


def run_demo(seed=17, output=None, *, fixture=None):
    """Fit a toy tabular model; exercise untrained neural models; optionally save."""
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("Nonnegative integer seed required")
    destination = Path(output) if output is not None else None
    if destination is not None and destination.exists():
        raise FileExistsError("Output directory already exists; choose a new directory")
    config = public_configuration()
    alerts, episodes, rows, records, grids = (
        build_fixture(config, seed) if fixture is None else fixture
    )
    assignment = split_patients(episodes, seed)
    partitions = {
        s: [r for r in rows if assignment[r["episode_id"]] == s]
        for s in ("train", "validation", "test")
    }
    subjects = {s: {r["patient"] for r in rs} for s, rs in partitions.items()}
    if any(
        subjects[a] & subjects[b]
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise ValueError("Subject leakage")
    kwargs = {
        "fitting_patients": subjects["train"],
        "forbidden_patients": subjects["validation"] | subjects["test"],
    }
    tabular = TabularBaseline(config, seed).fit(partitions["train"], **kwargs)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    models = {"grud": make_grud("grud", config).eval(), "mtan": make_mtan("mtan", config).eval()}
    preprocessors = {
        "grud": TemporalPreprocessor(config).fit(partitions["train"], records, **kwargs),
        "mtan": MTANPreprocessor(config).fit(partitions["train"], records, **kwargs),
    }
    scores = {"tabular": {s: tabular.predict_proba(partitions[s]) for s in ("validation", "test")}}
    for family, model in models.items():
        scores[family] = {}
        for split in ("validation", "test"):
            inputs = preprocessors[family].transform(partitions[split], records, family)
            with torch.inference_mode():
                scores[family][split] = torch.sigmoid(
                    model(**{k: torch.from_numpy(v) for k, v in inputs.items()})
                ).numpy()
    summary = {
        "synthetic": True,
        "manuscript_result_reproduction": False,
        "seed": seed,
        "fixture": "WHOLLY FICTIONAL; NOT CLINICAL DATA",
        "models": {
            "tabular": "logistic regression fitted on fictional training subjects",
            "grud": "untrained random initialization",
            "mtan": "untrained random initialization",
        },
        "episode_count": len(episodes),
        "subject_count": len({r["patient"] for r in rows}),
        "split_episode_counts": {s: len(rs) for s, rs in partitions.items()},
        "split_subject_counts": {s: len(ids) for s, ids in subjects.items()},
        "features": {
            "tabular_context": 56,
            "episode_statistics": 90,
            "temporal_slots": 288,
            "channels": 7,
        },
        "evaluation": {},
    }
    predictions = []
    for family, values in scores.items():
        validation_y = [int(r["label"] == 3) for r in partitions["validation"]]
        test_y = [int(r["label"] == 3) for r in partitions["test"]]
        chosen = choose_threshold(validation_y, values["validation"])
        summary["evaluation"][family] = {
            "threshold_selection_population": "fictional validation subjects",
            "threshold": chosen,
            "test_ranking": binary_metrics(test_y, values["test"]),
            "test_operating": operating_metrics(test_y, values["test"], chosen["threshold"]),
        }
        for split in ("validation", "test"):
            for row, score in zip(partitions[split], values[split], strict=True):
                predictions.append(
                    {
                        "synthetic": True,
                        "model": family,
                        "split": split,
                        "subject": row["patient"],
                        "episode_id": row["episode_id"],
                        "label": int(row["label"] == 3),
                        "score": float(score),
                    }
                )
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=False)
        for name, value in (
            ("metrics.json", summary),
            ("alerts.json", alerts),
            ("episodes.json", episodes),
            ("splits.json", assignment),
        ):
            (destination / name).write_text(
                json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
        with (destination / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
            writer.writeheader()
            writer.writerows(predictions)
        np.savez_compressed(destination / "fictional_wearables.npz", **grids)
        (destination / "README.txt").write_text(
            "WHOLLY FICTIONAL OUTPUT. No clinical results or fitted scientific checkpoints.\n"
            "Wearable arrays follow ai_cvd.data.synthetic.COLUMNS; one array per fictional episode.\n"
            "Neural scores come from untrained models. Do not interpret toy metrics clinically.\n",
            encoding="utf-8",
        )
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, default=Path("outputs/synthetic"))
    args = parser.parse_args(argv)
    result = run_demo(args.seed, args.output)
    print(
        f"Fictional demonstration complete: {result['episode_count']} episodes; output: {args.output}"
    )
    print("Toy metrics only; no published or held-out clinical results were reproduced.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
