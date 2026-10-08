import importlib

import numpy as np
import pytest
import torch

from ai_cvd.features.attention import MTANPreprocessor
from ai_cvd.features.temporal import TemporalPreprocessor
from ai_cvd.models.grud import make_model as make_grud
from ai_cvd.models.mtan import make_model as make_mtan
from ai_cvd.models.mtan import masked_attention


@pytest.mark.parametrize(
    "module",
    [
        "ai_cvd",
        "ai_cvd.demo",
        "ai_cvd.features.episode",
        "ai_cvd.models.tabular",
        "ai_cvd.models.lstm_vae",
        "ai_cvd.models.continuous",
        "ai_cvd.models.legacy.cgta",
        "ai_cvd.models.legacy.hybrid_cgta",
        "ai_cvd.models.legacy.supervised_sequence",
    ],
)
def test_package_imports(module):
    assert importlib.import_module(module).__file__


@pytest.mark.parametrize("family", ["grud", "relative_grud", "mtan"])
def test_temporal_forward_and_missing_placeholder_invariance(fixture, family):
    config, _, _, rows, records, _ = fixture
    cls = MTANPreprocessor if family == "mtan" else TemporalPreprocessor
    prep = cls(config).fit(rows[:8], records, fitting_patients={r["patient"] for r in rows[:8]})
    inputs = prep.transform(rows[8:10], records, family)
    assert inputs["context"].shape == (2, 56)
    assert inputs["masks"].any() and (~inputs["masks"].astype(bool)).any()
    model = (make_mtan if family == "mtan" else make_grud)(family, config).eval()
    tensors = {k: torch.from_numpy(v) for k, v in inputs.items()}
    with torch.inference_mode():
        first = model(**tensors)
        tensors["values"][tensors["masks"] == 0] = float("nan")
        second = model(**tensors)
    assert first.shape == (2,) and torch.isfinite(first).all()
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_empty_attention_is_finite_zero():
    q, k = torch.randn(2, 4, 48, 4), torch.randn(2, 4, 288, 4)
    values = torch.full((2, 288, 7), float("nan"))
    result, support, weights = masked_attention(q, k, values, torch.zeros_like(values))
    assert float(result.abs().sum() + support.sum() + weights.sum()) == 0


def test_mtan_rejects_alarm_time(fixture):
    config, _, _, rows, records, _ = fixture
    prep = MTANPreprocessor(config).fit(
        rows[:8], records, fitting_patients={r["patient"] for r in rows[:8]}
    )
    inputs = {
        k: torch.from_numpy(v) for k, v in prep.transform(rows[8:10], records, "mtan").items()
    }
    inputs["key_times"][:, -1] = 0
    with pytest.raises(ValueError):
        make_mtan("mtan", config)(**inputs)


def test_grud_decay_equations_and_cache():
    from ai_cvd.models.grud import GRUD

    model = GRUD(7)
    with torch.no_grad():
        model.hidden_decay.weight.fill_(0.1)
        model.hidden_decay.bias.zero_()
        model.gates.weight.zero_()
        model.gates.bias.zero_()
        model.candidate.weight.zero_()
        model.candidate.bias.zero_()
    value = torch.full((2, 7), float("nan"))
    mask = torch.zeros((2, 7))
    value[:, 0], mask[:, 0] = 4, 1
    h, cache, gx, gh = model.step(
        value, mask, torch.ones(2, 7), torch.ones(2, 7), torch.full((2, 7), 3.0), torch.ones(2, 32)
    )
    np.testing.assert_allclose(gx.detach(), np.exp(-1), rtol=1e-6)
    np.testing.assert_allclose(gh.detach(), np.exp(-0.7), rtol=1e-6)
    np.testing.assert_allclose(h.detach(), 0.5 * np.exp(-0.7), rtol=1e-6)
    assert (cache[:, 0] == 4).all() and (cache[:, 1:] == 3).all()


def test_lstm_vae_shape():
    from ai_cvd.models.lstm_vae import LSTM_VAE

    model = LSTM_VAE(input_dim=6, sequence_length=12, hidden_dim=8, embedding_dim=4).eval()
    reconstruction, mu, logvar, z = model(torch.zeros(2, 12, 6))
    assert reconstruction.shape == (2, 12, 6)
    assert mu.shape == logvar.shape == z.shape == (2, 4)


@pytest.mark.parametrize("recipe", ["R0", "R1"])
def test_continuous_masked_model(recipe):
    from ai_cvd.features.continuous import make_view
    from ai_cvd.models.continuous import FrozenRisk, MaskedModel, masked_huber

    values = torch.randn(2, 12, 6)
    observed = torch.ones_like(values, dtype=torch.bool)
    observed[:, 0] = False
    process = torch.zeros(2, 12, 25)
    artificial = torch.zeros_like(observed)
    artificial[:, 1] = True
    view = make_view(values, observed, process, artificial)
    model = MaskedModel(recipe)
    output = model(view)
    assert output.shape == values.shape
    assert torch.isfinite(masked_huber(output, view))
    risk = FrozenRisk(model)
    with pytest.raises(ValueError):
        risk(view)
    assert risk(make_view(values, observed, process)).shape == (2,)


def test_legacy_forward_interfaces():
    from ai_cvd.models.legacy.cgta import CGTANet
    from ai_cvd.models.legacy.hybrid_cgta import HybridCGTANet
    from ai_cvd.models.legacy.supervised_sequence import SupervisedSequenceNet

    dynamic, static, clock = torch.zeros(2, 12, 6), torch.zeros(2, 3), torch.zeros(2, 12, 2)
    assert CGTANet(6, 3, 2).eval()(dynamic, static, clock).shape == (2,)
    assert HybridCGTANet(6, 3, 2, 5).eval()(dynamic, static, clock, torch.zeros(2, 5)).shape == (2,)
    assert SupervisedSequenceNet(6, 3, sequence_length=12).eval()(dynamic, static).shape == (2,)
