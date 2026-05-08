"""Mixed-precision recommendations from sensitivity scores.

Given per-layer sensitivity, produces several recommendations partitioning
layers between FP32 (kept at full precision) and quantized (left at INT8).
The user picks the strategy that fits their speed/accuracy budget; the
report renders all three side by side.

Pure data transformation -- no ORT, no I/O. Imports only from sensitivity.
"""

from __future__ import annotations

from dataclasses import dataclass

from quantprobe.sensitivity import LayerSensitivity

# Cosine similarity thresholds for built-in strategies. A layer is kept at
# FP32 when its cosine drops *strictly below* the threshold (low cosine =
# bad quantization = needs sparing). Strict less-than is intentional so
# borderline layers are not spared by default.
STRATEGIES: dict[str, float] = {
    "aggressive": 0.7,
    "balanced": 0.9,
    "conservative": 0.95,
}


@dataclass(frozen=True)
class MixedPrecisionRecommendation:
    """One precision plan: which layers to keep FP32 vs quantize.

    Attributes:
        strategy_name: Human-readable label ("aggressive", "balanced",
            "conservative") that the report shows.
        cosine_threshold: The cosine similarity cutoff used to produce
            this recommendation. Layers strictly below this go to
            ``fp32_layers``.
        fp32_layers: Names of layers to leave at FP32 precision.
        quantized_layers: Names of layers to keep quantized (INT8).
    """

    strategy_name: str
    cosine_threshold: float
    fp32_layers: list[str]
    quantized_layers: list[str]


def recommend(
    layers: list[LayerSensitivity],
) -> list[MixedPrecisionRecommendation]:
    """Produce one MixedPrecisionRecommendation per built-in strategy.

    For each strategy, partitions ``layers`` by cosine similarity: any
    layer whose score is strictly below the strategy's threshold is
    placed in ``fp32_layers``; the rest go to ``quantized_layers``.

    An empty input yields one recommendation per strategy with both lists
    empty -- a valid (if useless) result that the report can still render.

    Args:
        layers: Per-layer sensitivity scores from sensitivity.analyse.

    Returns:
        One recommendation per entry in ``STRATEGIES``, in insertion order.
    """
    return [
        _recommend_for_strategy(name, threshold, layers) for name, threshold in STRATEGIES.items()
    ]


def _recommend_for_strategy(
    strategy_name: str,
    threshold: float,
    layers: list[LayerSensitivity],
) -> MixedPrecisionRecommendation:
    """Partition layers around a single threshold."""
    fp32_layers: list[str] = []
    quantized_layers: list[str] = []
    for layer in layers:
        if layer.cosine_similarity < threshold:
            fp32_layers.append(layer.name)
        else:
            quantized_layers.append(layer.name)
    return MixedPrecisionRecommendation(
        strategy_name=strategy_name,
        cosine_threshold=threshold,
        fp32_layers=fp32_layers,
        quantized_layers=quantized_layers,
    )
