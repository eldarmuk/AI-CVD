import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "release_checks", Path(__file__).parents[1] / "tools/release_checks.py"
)
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)


@pytest.mark.parametrize(
    "name",
    [
        "../escape.py",
        "/escape.py",
        "data/example.json",
        "model.pt",
        "table.csv",
        "output.npz",
        "notebooks/example.md",
    ],
)
def test_reject_unapproved_paths(name):
    assert checks.path_problem(name)


def test_review_manifest_detects_missing_added_and_changed_files(tmp_path):
    (tmp_path / "README.md").write_text("Reviewed\n", encoding="utf-8")
    manifest = {
        "files": [{"path": "README.md", "sha256": hashlib.sha256(b"Reviewed\n").hexdigest()}]
    }
    (tmp_path / checks.MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    files = ["README.md", checks.MANIFEST]
    assert checks.inspect(tmp_path, files) == []
    assert checks.inspect(tmp_path, files + ["new.py"])
    assert checks.inspect(tmp_path, [checks.MANIFEST])
    (tmp_path / "README.md").write_text("Changed\n", encoding="utf-8")
    assert checks.inspect(tmp_path, files)


def test_links_and_fragments(tmp_path):
    (tmp_path / "target.md").write_text("# Real heading\n", encoding="utf-8")
    assert not checks.link_problems(tmp_path, "README.md", "[ok](target.md#real-heading)")
    assert checks.link_problems(tmp_path, "README.md", "[bad](missing.md)")
    assert checks.link_problems(tmp_path, "README.md", "[bad](target.md#absent)")
