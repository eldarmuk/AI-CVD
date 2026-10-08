import copy

import numpy as np
import pandas as pd
import pytest

from ai_cvd.data.episodes import group_episodes, split_patients
from ai_cvd.data.synthetic import COLUMNS, generate_alerts, toy_grid
from ai_cvd.features.episode import STEP_NS, episode_statistics
from ai_cvd.features.temporal import elapsed, slot_starts


def test_fixed_seed_generation():
    alerts = generate_alerts()
    assert alerts == generate_alerts()
    a = toy_grid(alerts[0]["timestamp"], 0, 17)[0]
    np.testing.assert_array_equal(a, toy_grid(alerts[0]["timestamp"], 0, 17)[0])
    assert not np.array_equal(a, toy_grid(alerts[0]["timestamp"], 0, 18)[0], equal_nan=True)
    assert all(pd.Timestamp(x["timestamp"]).year == 2099 and x["synthetic"] for x in alerts)


def test_episode_grouping_chained_gap_and_utc_order():
    alerts = generate_alerts()[:2]
    third = dict(
        alerts[1], synthetic_alert_id="SYNTHETIC_ALERT_EXTRA", timestamp="2099-01-15T12:20:00+00:00"
    )
    alerts[1]["timestamp"] = "2099-01-15T13:10:00+01:00"
    result = group_episodes([third, *reversed(alerts)])
    assert len(result) == 1
    assert result[0]["constituent_count"] == 3
    assert result[0]["anchor"] == "2099-01-15T12:00:00Z"


@pytest.mark.parametrize(
    "mutation",
    [
        {"synthetic": False},
        {"synthetic_subject": "real-person"},
        {"timestamp": "2099-01-15"},
        {"synthetic_outcome_code": 1.5},
        {"synthetic_outcome_code": True},
        {"synthetic_alert_id": "private-id"},
    ],
)
def test_invalid_alert_rejected(mutation):
    alert = dict(generate_alerts()[0], **mutation)
    with pytest.raises(ValueError):
        group_episodes([alert])


def test_duplicate_alert_rejected():
    alert = generate_alerts()[0]
    with pytest.raises(ValueError):
        group_episodes([alert, alert])


def test_patient_split_isolation(fixture):
    _, _, episodes, *_ = fixture
    assignment = split_patients(episodes)
    assert assignment == split_patients(list(reversed(episodes)))
    sets = {
        s: {r["patient"] for r in episodes if assignment[r["episode_id"]] == s}
        for s in ("train", "validation", "test")
    }
    assert [len(sets[s]) for s in sets] == [8, 2, 2]
    assert not sets["train"] & (sets["validation"] | sets["test"])
    assert not sets["validation"] & sets["test"]
    assert len(assignment) == 24


@pytest.mark.parametrize(
    "anchor", ["2099-01-15T12:00:00Z", "2099-01-15T12:02:59Z", "2099-03-29T01:02:00Z"]
)
def test_strict_pre_alarm_slots(anchor):
    starts = slot_starts(anchor)
    assert len(starts) == 287
    assert np.all(starts + STEP_NS < pd.Timestamp(anchor).value)
    assert np.all(starts >= pd.Timestamp(anchor).value - 24 * 3600_000_000_000)


def test_future_and_alarm_bucket_cannot_change_features(fixture):
    config, _, episodes, _, _, _ = fixture
    episode = episodes[0]
    cfg = dict(config, _canonical_feature_names=list(COLUMNS))
    grid, start, contributors = toy_grid(episode["anchor"], 0)
    first = episode_statistics(episode["patient"], episode, grid, start, contributors, cfg)
    changed = grid.copy()
    starts = start + np.arange(len(grid)) * STEP_NS
    changed[starts + STEP_NS >= pd.Timestamp(episode["anchor"]).value] = 1e9
    second = episode_statistics(episode["patient"], episode, changed, start, contributors, cfg)
    np.testing.assert_allclose(
        list(first["features"].values()), list(second["features"].values()), equal_nan=True
    )
    assert first["recent_buckets"] == second["recent_buckets"]


def test_elapsed_time_and_padding():
    delta = elapsed(np.array([[0], [0], [1], [0], [0]]), np.array([False, True, True, True, True]))
    np.testing.assert_allclose(delta[:, 0], [0, 0, 1 / 12, 1 / 12, 2 / 12])


def test_partition_fit_rejects_heldout_and_transform_does_not_refit(fixture):
    from ai_cvd.features.episode import PartitionPreprocessor

    config, _, _, rows, _, _ = fixture
    fitting = rows[:8]
    ids = {r["patient"] for r in fitting}
    prep = PartitionPreprocessor(config)
    with pytest.raises(ValueError):
        prep.fit(fitting, fitting_patients=ids, forbidden_patients=ids)
    prep.fit(fitting, fitting_patients=ids)
    state = copy.deepcopy(prep.state())
    values = prep.transform(rows[8:], config["models"]["neural_context_subset"])
    assert values.shape == (16, 56) and np.isfinite(values).all()
    assert state == prep.state()
