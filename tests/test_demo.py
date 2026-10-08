import json
import subprocess
import sys

import numpy as np
import pytest

from ai_cvd.demo import run_demo


def test_complete_demo_and_reproducibility(tmp_path):
    output = tmp_path / "fictional"
    result = run_demo(output=output)
    assert result == run_demo()
    assert result["synthetic"] and not result["manuscript_result_reproduction"]
    assert result["episode_count"] == 24
    assert result["split_subject_counts"] == {"train": 8, "validation": 2, "test": 2}
    assert set(result["evaluation"]) == {"tabular", "grud", "mtan"}
    assert json.loads((output / "metrics.json").read_text()) == result
    assert (output / "predictions.csv").exists()
    with np.load(output / "fictional_wearables.npz", allow_pickle=False) as archive:
        assert len(archive.files) == 24
    with pytest.raises(FileExistsError):
        run_demo(output=output)


def test_cli_from_unrelated_directory(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "ai_cvd.demo", "--output", str(tmp_path / "demo")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Fictional demonstration complete" in result.stdout


def test_installed_configuration():
    from importlib.resources import files

    assert files("ai_cvd").joinpath("synthetic.toml").is_file()
