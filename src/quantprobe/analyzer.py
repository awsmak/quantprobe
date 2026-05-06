"""End-to-end orchestrator for the quantization sensitivity pipeline.

Composes quantizer, activation_collector, and sensitivity into a single
public function the CLI can call. Pure orchestration -- no file I/O
(handled by data_loader) and no rendering (handled by report).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import onnx
import onnx.shape_inference

from quantprobe.quantizer import QuantizationConfig, quantize
from quantprobe.sensitivity import LayerSensitivity
from quantprobe.sensitivity import analyse as run_sensitivity


@dataclass(frozen=True)
class AnalysisResult:
    """Full output of an analyse() run, ready for the report renderer.

    Attributes:
        layers: Per-layer sensitivity scores. Unsorted -- the report sorts
            by whichever metric it presents first.
        config: Quantization configuration that was applied.
        calibration_sample_count: Number of calibration samples that were
            fed into the quantizer. The report shows this so users can
            judge whether more calibration data might help.
        fp32_model_name: Human-readable identifier for the source model
            (graph name from the ONNX file).
        timestamp: When the analysis was finalized.
    """

    layers: list[LayerSensitivity]
    config: QuantizationConfig
    calibration_sample_count: int
    fp32_model_name: str
    timestamp: datetime


def analyse(
    fp32_model: onnx.ModelProto,
    calibration_data: np.ndarray,
    input_name: str,
    config: QuantizationConfig | None = None,
) -> AnalysisResult:
    """Quantize the model, score every layer, return a structured result.

    Args:
        fp32_model: Source FP32 model. Will be shape-inferred internally;
            caller does not need to pre-process it.
        calibration_data: Array of shape (N, *input_dims_without_batch).
            Used in full for quantizer calibration; only the first sample
            is used for the sensitivity forward pass (one inference per
            model is sufficient to compare activations).
        input_name: Name of the model input the calibration data feeds.
        config: Quantization configuration. Defaults to per-channel,
            symmetric, uint8 activations, int8 weights.

    Returns:
        AnalysisResult bundling the per-layer scores with the metadata
        the report needs (config, sample count, model name, timestamp).

    Raises:
        QuantizationError: Propagated from quantizer if quantization fails.
        SensitivityError: Propagated from sensitivity if activation
            collection or scoring fails.
    """
    if config is None:
        config = QuantizationConfig()

    fp32_inferred = onnx.shape_inference.infer_shapes(fp32_model)
    quant_model = quantize(fp32_inferred, calibration_data, input_name, config=config)
    quant_inferred = onnx.shape_inference.infer_shapes(quant_model)

    inputs = {input_name: calibration_data[0:1]}
    layers = run_sensitivity(fp32_inferred, quant_inferred, inputs)

    return AnalysisResult(
        layers=layers,
        config=config,
        calibration_sample_count=len(calibration_data),
        fp32_model_name=fp32_model.graph.name,
        timestamp=datetime.now(),
    )
