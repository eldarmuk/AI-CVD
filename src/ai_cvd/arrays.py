"""Explicit ID-bearing sequence export. No normalization, fitting or model selection."""
from itertools import groupby
import hashlib
import json
from pathlib import Path

from .cli import read_jsonl, verify_run, file_hash
from .dataset import sequence_for_sample, sample_id
from .features import FEATURE_NAMES


def export_sequences(run, output, task, split, kind="stream"):
    import numpy as np
    run, output = Path(run), Path(output)
    meta = verify_run(run, task)
    if split not in {"train", "validation", "test"}:
        raise ValueError("Explicit split required")
    if kind not in {"stream", "unsupervised_training"} or (kind == "unsupervised_training" and split != "train"):
        raise ValueError("Retrospective training exports must be training-only")
    manifest = run / ("stream_manifest.jsonl" if kind == "stream" else "unsupervised_training_manifest.jsonl")
    source_digest = file_hash(manifest)
    output.mkdir(parents=True, exist_ok=False)
    groups = iter(groupby(read_jsonl(manifest), key=lambda r: r["senior_id"]))
    current = next(groups, None)
    count, shards = 0, []
    seen = set()
    with open(output / "sample_manifest.jsonl", "x", encoding="utf-8") as audit:
        for sid, feature_group in groupby(read_jsonl(run / "features.jsonl"), key=lambda r: r["senior_id"]):
            if current is None or current[0] != sid:
                continue
            samples = list(current[1])
            current = next(groups, None)
            if not samples or samples[0]["split"] != split:
                continue
            features = list(feature_group)
            # Bound sequence expansion to 256 windows per shard, independent of patient size.
            for start in range(0, len(samples), 256):
                batch = samples[start:start+256]
                for row in batch:
                    if (row["sample_id"] != sample_id(row["senior_id"], row["prediction_time"]) or row["split"] != split
                            or row["endpoint"] != meta["endpoint"] or row["sample_id"] in seen):
                        raise ValueError("Sample identity/split mismatch")
                    seen.add(row["sample_id"])
                X = np.asarray([sequence_for_sample(features, row, task) for row in batch], dtype=np.float32)
                name = f"part_{len(shards):06d}.npz"
                np.savez_compressed(output / name, X=X, y=np.asarray([r["target"] for r in batch], dtype=np.int8),
                                    sample_ids=np.asarray([r["sample_id"] for r in batch]),
                                    task_identifier=np.asarray(task.identifier), endpoint=np.asarray(meta["endpoint"]),
                                    source_manifest_sha256=np.asarray(source_digest), feature_names=np.asarray(FEATURE_NAMES))
                for index, row in enumerate(batch):
                    audit.write(json.dumps(dict(row, array_file=name, array_row=index,
                                               source_manifest_sha256=source_digest,
                                               input_sha256=hashlib.sha256(X[index].tobytes()).hexdigest())) + "\n")
                shards.append({"name": name, "sha256": file_hash(output/name), "samples": len(batch)})
                count += len(batch)
    if current is not None:
        raise ValueError("Manifest patient cannot be joined to feature records")
    result = {"task_identifier": task.identifier, "endpoint": meta["endpoint"], "feature_names": FEATURE_NAMES,
              "split": split, "sampling": kind, "samples": count, "shards": shards,
              "source_manifest_sha256": source_digest, "sample_manifest_sha256": file_hash(output/"sample_manifest.jsonl")}
    with open(output/"array_metadata.json", "x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result


def verified_shards(output, task):
    """Required reader for ID-bearing arrays; detects reordered values, IDs and targets.

    Yields dictionaries with X, y and sample_ids together. Never join predictions by
    row position: use dataset.align_by_sample_id with the task and endpoint.
    """
    import numpy as np
    output = Path(output)
    meta = json.loads((output / "array_metadata.json").read_text(encoding="utf-8"))
    if meta["task_identifier"] != task.identifier or tuple(meta["feature_names"]) != FEATURE_NAMES:
        raise ValueError("Array task/schema mismatch")
    if file_hash(output / "sample_manifest.jsonl") != meta["sample_manifest_sha256"]:
        raise ValueError("Array manifest checksum mismatch")
    groups = iter(groupby(read_jsonl(output / "sample_manifest.jsonl"), lambda r: r["array_file"]))
    seen, count = set(), 0
    for shard in meta["shards"]:
        name = shard["name"]
        if Path(name).name != name or file_hash(output / name) != shard["sha256"]:
            raise ValueError("Array shard checksum/path mismatch")
        group = next(groups, None)
        if group is None or group[0] != name:
            raise ValueError("Array manifest/shard mismatch")
        rows = list(group[1])
        with np.load(output / name, allow_pickle=False) as data:
            values = {key: data[key] for key in data.files}
        n = len(rows)
        if (n != shard["samples"] or values["X"].shape != (n, task.sequence_steps, len(FEATURE_NAMES))
                or values["y"].shape != (n,) or values["sample_ids"].shape != (n,)
                or str(values["task_identifier"]) != task.identifier or str(values["endpoint"]) != meta["endpoint"]
                or str(values["source_manifest_sha256"]) != meta["source_manifest_sha256"]
                or tuple(values["feature_names"]) != FEATURE_NAMES):
            raise ValueError("Array dimensions/schema mismatch")
        for i, row in enumerate(rows):
            key = row["sample_id"]
            if (key in seen or key != sample_id(row["senior_id"], row["prediction_time"])
                    or row["array_row"] != i or values["sample_ids"][i] != key
                    or values["y"][i] != row["target"] or row["split"] != meta["split"]
                    or row["endpoint"] != meta["endpoint"] or row["task_identifier"] != task.identifier
                    or row["source_manifest_sha256"] != meta["source_manifest_sha256"]
                    or hashlib.sha256(values["X"][i].tobytes()).hexdigest() != row["input_sha256"]):
                raise ValueError("Array sample identity/content mismatch")
            seen.add(key)
        count += n
        yield values
    if next(groups, None) is not None or count != meta["samples"]:
        raise ValueError("Array sample count mismatch")
