"""Tests for :mod:`quantprobe.mixed_precision`.

Run a single class with::

    pytest tests/test_mixed_precision.py::TestRecommendThresholding -v
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from quantprobe.mixed_precision import (
    STRATEGIES,
    MixedPrecisionRecommendation,
    recommend,
)
from quantprobe.sensitivity import LayerSensitivity


def _make_layer(name: str, cosine: float) -> LayerSensitivity:
    """Build a LayerSensitivity with a given cosine score; other fields are dummy."""
    return LayerSensitivity(
        name=name,
        cosine_similarity=cosine,
        mse=0.0,
        max_abs_error=0.0,
        snr_db=20.0,
    )


class TestMixedPrecisionRecommendation:
    """Public dataclass contract."""

    def test_construct_with_all_fields(self) -> None:
        rec = MixedPrecisionRecommendation(
            strategy_name="balanced",
            cosine_threshold=0.9,
            fp32_layers=["a", "b"],
            quantized_layers=["c"],
        )
        assert rec.strategy_name == "balanced"
        assert rec.cosine_threshold == 0.9
        assert rec.fp32_layers == ["a", "b"]
        assert rec.quantized_layers == ["c"]

    def test_is_immutable(self) -> None:
        rec = MixedPrecisionRecommendation(
            strategy_name="x",
            cosine_threshold=0.5,
            fp32_layers=[],
            quantized_layers=[],
        )
        with pytest.raises(FrozenInstanceError):
            rec.strategy_name = "y"  # type: ignore[misc]


class TestStrategiesConstant:
    """STRATEGIES is the public list of named thresholds."""

    def test_has_three_strategies(self) -> None:
        assert set(STRATEGIES.keys()) == {"aggressive", "balanced", "conservative"}

    def test_thresholds_ordered_by_aggressiveness(self) -> None:
        # Aggressive saves fewest layers (lowest threshold);
        # conservative saves most (highest threshold).
        assert STRATEGIES["aggressive"] < STRATEGIES["balanced"]
        assert STRATEGIES["balanced"] < STRATEGIES["conservative"]

    def test_thresholds_in_valid_range(self) -> None:
        # Cosine similarity is bounded [-1, 1]; thresholds in (0, 1) make sense.
        for value in STRATEGIES.values():
            assert 0.0 < value < 1.0


class TestRecommendBasics:
    """recommend() returns one entry per strategy, correctly named."""

    def test_returns_list(self) -> None:
        result = recommend([_make_layer("a", 0.99)])
        assert isinstance(result, list)

    def test_one_recommendation_per_strategy(self) -> None:
        result = recommend([_make_layer("a", 0.99)])
        names = {rec.strategy_name for rec in result}
        assert names == set(STRATEGIES.keys())

    def test_recommendation_carries_threshold(self) -> None:
        result = recommend([_make_layer("a", 0.99)])
        for rec in result:
            assert rec.cosine_threshold == STRATEGIES[rec.strategy_name]


class TestRecommendThresholding:
    """Below threshold -> FP32 (keep). At-or-above threshold -> quantized."""

    def test_low_cosine_goes_to_fp32(self) -> None:
        # Cosine 0.5 is below all three thresholds (0.7, 0.9, 0.95) -- a
        # broken layer should be kept FP32 by every strategy.
        layers = [_make_layer("bad_layer", 0.5)]
        result = recommend(layers)
        for rec in result:
            assert "bad_layer" in rec.fp32_layers
            assert "bad_layer" not in rec.quantized_layers

    def test_high_cosine_goes_to_quantized(self) -> None:
        # Cosine 0.99 is above all thresholds -- a healthy layer should be
        # quantized by every strategy.
        layers = [_make_layer("good_layer", 0.99)]
        result = recommend(layers)
        for rec in result:
            assert "good_layer" in rec.quantized_layers
            assert "good_layer" not in rec.fp32_layers

    def test_layer_at_exact_threshold_goes_to_quantized(self) -> None:
        # Strict less-than: cosine == threshold is NOT below, so quantized.
        # This is a deliberate choice to avoid sparing borderline layers.
        layers = [_make_layer("borderline", 0.9)]
        result = recommend(layers)
        balanced = next(r for r in result if r.strategy_name == "balanced")
        assert "borderline" in balanced.quantized_layers


class TestRecommendStrategiesDifferentiate:
    """Aggressive saves fewer layers; conservative saves more."""

    def test_fp32_sets_are_nested(self) -> None:
        layers = [
            _make_layer("a", 0.5),  # below all thresholds
            _make_layer("b", 0.8),  # below balanced & conservative
            _make_layer("c", 0.92),  # below conservative only
            _make_layer("d", 0.99),  # above all thresholds
        ]
        result = {r.strategy_name: set(r.fp32_layers) for r in recommend(layers)}
        # Nested subset: aggressive saves a strict subset of what balanced
        # saves, which saves a subset of what conservative saves.
        assert result["aggressive"] <= result["balanced"]
        assert result["balanced"] <= result["conservative"]

    def test_aggressive_only_saves_truly_broken(self) -> None:
        layers = [
            _make_layer("really_bad", 0.5),
            _make_layer("kinda_bad", 0.8),
            _make_layer("fine", 0.99),
        ]
        result = recommend(layers)
        aggressive = next(r for r in result if r.strategy_name == "aggressive")
        assert aggressive.fp32_layers == ["really_bad"]
        # 0.8 is above 0.7, so kinda_bad is quantized in aggressive.
        assert "kinda_bad" in aggressive.quantized_layers


class TestRecommendEdgeCases:
    """Empty and uniform inputs must not crash."""

    def test_empty_input_returns_recommendation_per_strategy(self) -> None:
        result = recommend([])
        assert len(result) == len(STRATEGIES)
        for rec in result:
            assert rec.fp32_layers == []
            assert rec.quantized_layers == []

    def test_all_layers_below_aggressive_all_fp32(self) -> None:
        # Cosine 0.1 is below every threshold; every strategy spares everything.
        layers = [_make_layer(name, 0.1) for name in ["a", "b", "c"]]
        result = recommend(layers)
        for rec in result:
            assert set(rec.fp32_layers) == {"a", "b", "c"}
            assert rec.quantized_layers == []

    def test_all_layers_above_conservative_none_fp32(self) -> None:
        layers = [_make_layer(name, 0.99) for name in ["a", "b", "c"]]
        result = recommend(layers)
        for rec in result:
            assert rec.fp32_layers == []
            assert set(rec.quantized_layers) == {"a", "b", "c"}
