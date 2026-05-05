"""Tests for :mod:`quantprobe.sensitivity`.

These tests exercise the comparison and ranking logic in isolation by
mocking activation_collector.collect to return controlled activation
dicts. End-to-end integration with real ONNX models is covered by the
analyzer tests.

Run a single class with::

    pytest tests/test_sensitivity.py::TestAnalyseHappyPath -v
"""

from __future__ import annotations

from typing import Callable
from unittest.mock import patch

import numpy as np
import onnx
import pytest

from quantprobe.sensitivity import LayerSensitivity, analyse


def _make_collect_stub(
    fp32_acts: dict[str, np.ndarray],
    quant_acts: dict[str, np.ndarray],
) -> Callable[..., dict[str, np.ndarray]]:
    """Build a stub for activation_collector.collect.

    Returns fp32_acts on the first call, quant_acts on the second. analyse()
    is contractually required to call collect with the FP32 model first.
    """
    calls = {"count": 0}

    def stub(model: onnx.ModelProto, inputs: dict[str, np.ndarray]):
        calls["count"] += 1
        return fp32_acts if calls["count"] == 1 else quant_acts

    return stub


# A dummy ModelProto suffices because collect is mocked -- analyse never
# actually inspects the model when the stub is in place.
_DUMMY_MODEL = onnx.ModelProto()
_DUMMY_INPUTS: dict[str, np.ndarray] = {"input": np.zeros((1, 4), dtype=np.float32)}


class TestAnalyseHappyPath:
    """analyse returns one LayerSensitivity per shared layer."""

    def test_returns_list_of_layer_sensitivity(self) -> None:
        fp32 = {"layer_a": np.array([1.0, 2.0, 3.0], dtype=np.float32)}
        quant = {"layer_a": np.array([1.1, 2.1, 2.9], dtype=np.float32)}
        with patch("quantprobe.sensitivity.collect", _make_collect_stub(fp32, quant)):
            results = analyse(_DUMMY_MODEL, _DUMMY_MODEL, _DUMMY_INPUTS)
        assert isinstance(results, list)
        assert all(isinstance(r, LayerSensitivity) for r in results)

    def test_one_result_per_shared_layer(self) -> None:
        fp32 = {
            "layer_a": np.array([1.0, 2.0], dtype=np.float32),
            "layer_b": np.array([3.0, 4.0], dtype=np.float32),
            "layer_c": np.array([5.0, 6.0], dtype=np.float32),
        }
        quant = {
            "layer_a": np.array([1.1, 2.1], dtype=np.float32),
            "layer_b": np.array([3.1, 4.1], dtype=np.float32),
            "layer_c": np.array([5.1, 6.1], dtype=np.float32),
        }
        with patch("quantprobe.sensitivity.collect", _make_collect_stub(fp32, quant)):
            results = analyse(_DUMMY_MODEL, _DUMMY_MODEL, _DUMMY_INPUTS)
        assert len(results) == 3
        names = {r.name for r in results}
        assert names == {"layer_a", "layer_b", "layer_c"}

    def test_all_four_metrics_populated(self) -> None:
        fp32 = {"layer_a": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)}
        quant = {"layer_a": np.array([1.1, 2.1, 2.9, 4.1], dtype=np.float32)}
        with patch("quantprobe.sensitivity.collect", _make_collect_stub(fp32, quant)):
            results = analyse(_DUMMY_MODEL, _DUMMY_MODEL, _DUMMY_INPUTS)
        result = results[0]
        # All four metrics must be finite floats, not None or nan.
        assert isinstance(result.cosine_similarity, float)
        assert isinstance(result.mse, float)
        assert isinstance(result.max_abs_error, float)
        assert isinstance(result.snr_db, float)


class TestAnalyseMetricCorrectness:
    """Each LayerSensitivity field must hold the value its metric computes."""

    def test_identical_activations_give_perfect_scores(self) -> None:
        # Identical inputs: cosine = 1, mse = 0, max_abs = 0, snr = inf.
        acts = {"layer_a": np.array([1.0, 2.0, 3.0], dtype=np.float32)}
        with patch("quantprobe.sensitivity.collect", _make_collect_stub(acts, acts)):
            results = analyse(_DUMMY_MODEL, _DUMMY_MODEL, _DUMMY_INPUTS)
        r = results[0]
        assert r.cosine_similarity == pytest.approx(1.0)
        assert r.mse == pytest.approx(0.0)
        assert r.max_abs_error == pytest.approx(0.0)
        assert r.snr_db == np.inf

    def test_known_difference_gives_known_mse(self) -> None:
        fp32 = {"layer_a": np.array([1.0, 2.0, 3.0], dtype=np.float32)}
        quant = {"layer_a": np.array([2.0, 3.0, 4.0], dtype=np.float32)}
        with patch("quantprobe.sensitivity.collect", _make_collect_stub(fp32, quant)):
            results = analyse(_DUMMY_MODEL, _DUMMY_MODEL, _DUMMY_INPUTS)
        # All elementwise diffs are 1.0 -> mse = 1.0.
        assert results[0].mse == pytest.approx(1.0)


class TestLayerMatching:
    """Layers present in only one model must be excluded from results."""

    def test_layer_only_in_fp32_is_skipped(self) -> None:
        fp32 = {
            "shared": np.array([1.0, 2.0], dtype=np.float32),
            "fp32_only": np.array([3.0, 4.0], dtype=np.float32),
        }
        quant = {"shared": np.array([1.1, 2.1], dtype=np.float32)}
        with patch("quantprobe.sensitivity.collect", _make_collect_stub(fp32, quant)):
            results = analyse(_DUMMY_MODEL, _DUMMY_MODEL, _DUMMY_INPUTS)
        names = {r.name for r in results}
        assert names == {"shared"}

    def test_layer_only_in_quant_is_skipped(self) -> None:
        fp32 = {"shared": np.array([1.0, 2.0], dtype=np.float32)}
        quant = {
            "shared": np.array([1.1, 2.1], dtype=np.float32),
            "quant_only": np.array([3.0, 4.0], dtype=np.float32),
        }
        with patch("quantprobe.sensitivity.collect", _make_collect_stub(fp32, quant)):
            results = analyse(_DUMMY_MODEL, _DUMMY_MODEL, _DUMMY_INPUTS)
        names = {r.name for r in results}
        assert names == {"shared"}


class TestDegenerateLayerHandling:
    """Degenerate layers are skipped so one bad layer does not kill the run."""

    def test_zero_norm_layer_is_skipped(self) -> None:
        # layer_a has zero-norm fp32 activation -> cosine_similarity raises.
        # analyse must skip it but still report the healthy layer.
        fp32 = {
            "dead_layer": np.zeros(4, dtype=np.float32),
            "healthy": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        }
        quant = {
            "dead_layer": np.array([0.1, 0.2, 0.0, 0.0], dtype=np.float32),
            "healthy": np.array([1.1, 2.1, 2.9, 4.1], dtype=np.float32),
        }
        with patch("quantprobe.sensitivity.collect", _make_collect_stub(fp32, quant)):
            results = analyse(_DUMMY_MODEL, _DUMMY_MODEL, _DUMMY_INPUTS)
        names = {r.name for r in results}
        assert "dead_layer" not in names
        assert "healthy" in names

    def test_all_layers_degenerate_returns_empty_list(self) -> None:
        fp32 = {"a": np.zeros(3, dtype=np.float32), "b": np.zeros(3, dtype=np.float32)}
        quant = {"a": np.zeros(3, dtype=np.float32), "b": np.zeros(3, dtype=np.float32)}
        with patch("quantprobe.sensitivity.collect", _make_collect_stub(fp32, quant)):
            results = analyse(_DUMMY_MODEL, _DUMMY_MODEL, _DUMMY_INPUTS)
        assert results == []


class TestLayerSensitivityDataclass:
    """LayerSensitivity is the public data structure -- exercise its contract."""

    def test_is_immutable(self) -> None:
        # frozen=True raises FrozenInstanceError on attribute assignment.
        from dataclasses import FrozenInstanceError

        ls = LayerSensitivity(
            name="x", cosine_similarity=1.0, mse=0.0, max_abs_error=0.0, snr_db=42.0
        )
        with pytest.raises(FrozenInstanceError):
            ls.name = "y"  # type: ignore[misc]

    def test_fields_are_accessible_by_name(self) -> None:
        ls = LayerSensitivity(
            name="conv1", cosine_similarity=0.99, mse=0.01, max_abs_error=0.1, snr_db=20.0
        )
        assert ls.name == "conv1"
        assert ls.cosine_similarity == 0.99
        assert ls.mse == 0.01
        assert ls.max_abs_error == 0.1
        assert ls.snr_db == 20.0
