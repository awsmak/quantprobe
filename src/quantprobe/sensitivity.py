"""Per-layer quantization sensitivity analysis.

Calls activation_collector for both the FP32 and quantized models,
compares activations layer-by-layer using the four metrics, and returns
a structured result the caller can rank or render.

This is a per-layer approximation, not end-to-end sensitivity. Error
cancellation effects (e.g. across residual connections) are not captured;
see the project README for the tradeoff discussion.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import onnx

from quantprobe.activation_collector import collect
from quantprobe.exceptions import MetricsError
from quantprobe.metrics import cosine_similarity, max_abs_error, mse, snr_db

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LayerSensitivity:
    """Sensitivity scores for a single layer.

    Frozen so instances cannot be mutated downstream (the report and
    mixed_precision modules treat these as read-only data).

    Attributes:
        name: ONNX tensor name of the layer's output.
        cosine_similarity: 1.0 = identical direction, -1.0 = opposite.
        mse: Mean squared error between FP32 and quantized activations.
        max_abs_error: Largest single-element absolute difference.
        snr_db: Signal-to-noise ratio in decibels (np.inf if identical).
    """

    name: str
    cosine_similarity: float
    mse: float
    max_abs_error: float
    snr_db: float


def analyse(
    fp32_model: onnx.ModelProto,
    quant_model: onnx.ModelProto,
    inputs: dict[str, np.ndarray],
) -> list[LayerSensitivity]:
    """Compute per-layer sensitivity between an FP32 and a quantized model.

    Both models must already be shape-inferred. Layers present in only one
    model are excluded (defensive intersection -- they should match if both
    were built from the same source). Layers whose metrics are undefined
    (e.g. dead ReLUs with zero-norm activations) are skipped with a warning
    so a single bad layer does not abort the whole analysis.

    Args:
        fp32_model: Shape-inferred FP32 reference model.
        quant_model: Shape-inferred quantized model with the same graph
            structure.
        inputs: Calibration input tensors keyed by input name.

    Returns:
        Unsorted list of LayerSensitivity, one per healthy shared layer.
        Caller is responsible for sorting by whichever metric matters.

    Raises:
        SensitivityError: Propagated from activation_collector if either
            model fails to run inference.
    """
    fp32_acts = collect(fp32_model, inputs)
    quant_acts = collect(quant_model, inputs)

    shared_layers = sorted(set(fp32_acts.keys()) & set(quant_acts.keys()))

    results: list[LayerSensitivity] = []
    for name in shared_layers:
        try:
            scored = _score_layer(name, fp32_acts[name], quant_acts[name])
        except MetricsError as exc:
            logger.warning("Skipping layer %s: %s", name, exc)
            continue
        results.append(scored)

    return results


def _score_layer(name: str, fp32: np.ndarray, quant: np.ndarray) -> LayerSensitivity:
    """Compute all four metrics for one layer."""
    return LayerSensitivity(
        name=name,
        cosine_similarity=cosine_similarity(fp32, quant),
        mse=mse(fp32, quant),
        max_abs_error=max_abs_error(fp32, quant),
        snr_db=snr_db(fp32, quant),
    )
