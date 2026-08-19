import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


PIPELINE_PATH = Path(__file__).parent / "src" / "pipelines" / "01_generate_dataset.py"


def load_dataset_pipeline():
    spec = importlib.util.spec_from_file_location(
        "src.pipelines.generate_dataset_under_test",
        PIPELINE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_senior_frame(senior_id, n_rows, positive_target_idx=None, feature_offset=0):
    timestamps = pd.date_range("2026-01-01", periods=n_rows, freq="15min")
    labels = np.zeros(n_rows, dtype=int)
    if positive_target_idx is not None:
        labels[positive_target_idx] = 1

    return pd.DataFrame({
        "senior_id": senior_id,
        "timestamp": timestamps,
        "feature_a": np.arange(n_rows, dtype=np.float32) + feature_offset,
        "label_1": np.zeros(n_rows, dtype=int),
        "label_2": np.zeros(n_rows, dtype=int),
        "label_3": labels,
    })


def test_split_overlap_is_rejected():
    pipeline = load_dataset_pipeline()

    try:
        pipeline.assert_disjoint_splits({
            "train": [1, 2],
            "val": [3],
            "test": [2, 4],
        })
    except ValueError as exc:
        assert "split leakage" in str(exc)
    else:
        raise AssertionError("Expected overlap between train and test to be rejected")


def test_positive_target_row_is_outside_lookback_window():
    pipeline = load_dataset_pipeline()
    df = pd.concat([
        make_senior_frame(1, n_rows=7, positive_target_idx=4),
        make_senior_frame(2, n_rows=5, feature_offset=100),
    ], ignore_index=True)

    X, y = pipeline.generate_testing_data(
        df,
        test_seniors=[1, 2],
        seq_len=4,
        feature_cols=["feature_a"],
        split_label="UNIT",
    )

    assert sorted(y.tolist()) == [0, 1]
    positive_window = X[y == 1][0, :, 0]
    assert positive_window.tolist() == [0.0, 1.0, 2.0, 3.0]
    assert 4.0 not in positive_window


def test_windows_do_not_bridge_seniors():
    pipeline = load_dataset_pipeline()
    df = pd.concat([
        make_senior_frame(1, n_rows=3, feature_offset=0),
        make_senior_frame(2, n_rows=3, positive_target_idx=2, feature_offset=100),
    ], ignore_index=True)

    X, y = pipeline.generate_testing_data(
        df,
        test_seniors=[1, 2],
        seq_len=2,
        feature_cols=["feature_a"],
        split_label="UNIT",
    )

    assert sorted(y.tolist()) == [0, 1]
    assert X[y == 1][0, :, 0].tolist() == [100.0, 101.0]


if __name__ == "__main__":
    test_split_overlap_is_rejected()
    test_positive_target_row_is_outside_lookback_window()
    test_windows_do_not_bridge_seniors()
    print("All leakage regression tests passed.")
