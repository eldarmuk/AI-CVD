import numpy as np
import pytest

from ai_cvd.evaluation.metrics import binary_metrics, choose_threshold, operating_metrics


def test_known_metrics():
    result = binary_metrics([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    assert result["auroc"] == 0.75
    assert result["brier"] == pytest.approx(0.158125)
    assert result["ap"] == pytest.approx(5 / 6)


def test_whole_ties_are_never_split():
    chosen = choose_threshold([1, 0, 1, 0], [0.8, 0.8, 0.2, 0.1], target=1)
    assert chosen["threshold"] == 0.2
    assert chosen["TP"] == 2 and chosen["FP"] == 1
    assert operating_metrics([1, 0], [0.5, 0.5], 0.5)["fraction_prioritized"] == 1


def test_largest_threshold_breaks_equal_specificity():
    chosen = choose_threshold([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1], target=0.5)
    assert chosen["threshold"] == 0.9


def test_single_class():
    assert binary_metrics([0, 0], [0.2, 0.3])["auroc"] is None
    assert binary_metrics([0, 0], [0.2, 0.3])["ap"] is None
    assert not choose_threshold([1, 1], [0.2, 0.3])["available"]


@pytest.mark.parametrize(
    "y,p",
    [
        ([], []),
        ([0, 1], [0.1]),
        ([0, 2], [0.1, 0.2]),
        ([0, 1], [np.nan, 0.2]),
        ([0, 1], [-0.1, 0.2]),
        ([0, 1], [0.1, 1.2]),
        ([[0, 1]], [[0.1, 0.2]]),
    ],
)
def test_invalid_metrics(y, p):
    with pytest.raises(ValueError):
        binary_metrics(y, p)


@pytest.mark.parametrize("target", [0, -1, 1.1, float("nan")])
def test_invalid_threshold_target(target):
    with pytest.raises(ValueError):
        choose_threshold([0, 1], [0.1, 0.9], target)
