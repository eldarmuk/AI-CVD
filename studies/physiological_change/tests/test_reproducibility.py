import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hr_change.reproducibility import (
    DEFAULT_CONFIG,
    REPOSITORY_ROOT,
    STUDY_ROOT,
    _source_hashes,
    run_synthetic,
    sha256,
    verify_run,
)


class ReproducibilityTests(unittest.TestCase):
    def setUp(self):
        # Keep generated fixtures inside the ignored workspace output tree.
        self.temp_root = (REPOSITORY_ROOT / "runs" / "physiological-change" / "test-temp").resolve()
        if not self.temp_root.is_relative_to(REPOSITORY_ROOT.resolve()):
            raise ValueError("test cleanup path escaped the workspace")
        self.temp_root.mkdir(parents=True, exist_ok=True)

    def test_two_runs_have_identical_data_and_metrics(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as temporary:
            first = Path(temporary) / "first"
            second = Path(temporary) / "second"
            first_manifest = run_synthetic(first)
            second_manifest = run_synthetic(second)
            self.assertEqual(
                first_manifest["artifacts_sha256"], second_manifest["artifacts_sha256"]
            )
            self.assertTrue(verify_run(first)["verified"])

    def test_refuses_to_overwrite_even_an_empty_directory(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as temporary:
            with self.assertRaises(FileExistsError):
                run_synthetic(Path(temporary))

    def test_modified_artifact_fails_hash_verification(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as temporary:
            output = Path(temporary) / "run"
            run_synthetic(output)
            (output / "results.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_run(output)

    def test_rehashing_wrong_metrics_cannot_bypass_semantic_replay(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as temporary:
            output = Path(temporary) / "run"
            run_synthetic(output)
            result_path = output / "results.json"
            results = json.loads(result_path.read_text(encoding="utf-8"))
            results["methods"]["last_observation"]["counts"]["true_changes"] = 0
            result_path.write_text(json.dumps(results), encoding="utf-8")
            manifest_path = output / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifacts_sha256"]["results.json"] = sha256(result_path)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "semantic replay"):
                verify_run(output)

    def test_manifest_path_injection_rejected(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as temporary:
            output = Path(temporary) / "run"
            run_synthetic(output)
            manifest_path = output / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifacts_sha256"]["../outside.json"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "inventory"):
                verify_run(output)

    def test_malformed_manifest_types_fail_cleanly(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as temporary:
            output = Path(temporary) / "run"
            original = run_synthetic(output)
            malformed = [
                [],
                {**original, "schema_version": True},
                {**original, "schema_version": 1.0},
                {**original, "fixture_version": "unknown"},
                {**original, "artifacts_sha256": list(original["artifacts_sha256"])},
                {**original, "sources_sha256": []},
                {
                    **original,
                    "artifacts_sha256": {**original["artifacts_sha256"], "config.toml": None},
                },
            ]
            for payload in malformed:
                with self.subTest(payload=payload):
                    (output / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        verify_run(output)

    def test_invalid_configuration_does_not_create_output(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as temporary:
            root = Path(temporary)
            config_path = root / "invalid.toml"
            config_path.write_text(
                DEFAULT_CONFIG.read_text(encoding="utf-8").replace("10.0", "nan"), encoding="utf-8"
            )
            output = root / "run"
            with self.assertRaises(ValueError):
                run_synthetic(output, config_path)
            self.assertFalse(output.exists())

    def test_tracked_study_directory_cannot_receive_run_artifacts(self):
        forbidden = STUDY_ROOT / "must-not-create-synthetic-run"
        self.assertFalse(forbidden.exists())
        with self.assertRaisesRegex(ValueError, "outside the tracked study"):
            run_synthetic(forbidden)
        self.assertFalse(forbidden.exists())

    def test_source_snapshot_includes_nested_code_configuration_and_lock(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as temporary:
            repository = Path(temporary)
            study = repository / "studies" / "fixture"
            nested = study / "hr_change" / "adapters"
            nested.mkdir(parents=True)
            (nested / "loader.py").write_text("VERSION = 1\n", encoding="utf-8")
            (study / "configs").mkdir()
            (study / "configs" / "example.toml").write_text("version = 1\n", encoding="utf-8")
            (study / "uv.lock").write_text("version = 1\n", encoding="utf-8")
            snapshot = _source_hashes(study_root=study, repository_root=repository)
            self.assertEqual(
                set(snapshot),
                {
                    "studies/fixture/hr_change/adapters/loader.py",
                    "studies/fixture/configs/example.toml",
                    "studies/fixture/uv.lock",
                },
            )

    def test_hash_replay_verification_does_not_authenticate_git_claim(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as temporary:
            output = Path(temporary) / "run"
            manifest = run_synthetic(output)
            self.assertEqual(
                manifest["git"]["worktree_scope"], "entire_repository_including_untracked_files"
            )
            result = verify_run(output)
            self.assertTrue(result["verified"])
            self.assertFalse(result["git_revision_claim_verified"])


if __name__ == "__main__":
    unittest.main()
