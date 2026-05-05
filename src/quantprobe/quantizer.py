"""Static INT8 quantization wrapper around ORT's quantize_static.

INT4 weight quantization is intentionally not supported in v1. ORT's
public quantize_static accepts only 8-bit and 16-bit weight types; INT4
requires either MatMul4BitsQuantizer (LLM workloads, MatMul/Gather only)
or a vendor SDK (Qualcomm QNN, TensorRT) for Conv-INT4 on NPUs.
Tracked for v2.

Only onnxruntime is imported here, alongside model_runner.py and
activation_collector.py. Other modules must not import it.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)

from quantprobe.exceptions import QuantizationError

_VALID_8BIT_TYPES = frozenset({"int8", "uint8"})
_INT4_TYPES = frozenset({"int4", "uint4"})

_TYPE_TO_QUANTTYPE = {
    "int8": QuantType.QInt8,
    "uint8": QuantType.QUInt8,
}


@dataclass(frozen=True)
class QuantizationConfig:
    """Static quantization configuration.

    Frozen so instances cannot be mutated after validation. All values are
    checked in __post_init__ -- by the time you have a QuantizationConfig
    instance, you can trust its fields without re-validating.

    Attributes:
        per_channel: True for per-channel weight quantization. More
            accurate but some embedded NPUs require False.
        symmetric: True for symmetric quantization (zero-point=0).
            Required by some hardware (Qualcomm HTP).
        activation_type: One of "int8", "uint8".
        weight_type: One of "int8", "uint8".
    """

    per_channel: bool = True
    symmetric: bool = True
    activation_type: str = "uint8"
    weight_type: str = "int8"

    def __post_init__(self) -> None:
        for field_name, value in (
            ("activation_type", self.activation_type),
            ("weight_type", self.weight_type),
        ):
            if value in _INT4_TYPES:
                raise QuantizationError(
                    f"{field_name}={value!r} is not supported in v1. "
                    "ORT's public quantize_static does not expose INT4 weights. "
                    "For INT4 MatMul (LLM workloads), use ORT's MatMul4BitsQuantizer "
                    "directly. For INT4 Conv (vision NPUs like Qualcomm HTP), use "
                    "the vendor SDK (QNN, TensorRT). Tracked for v2."
                )
            if value not in _VALID_8BIT_TYPES:
                raise QuantizationError(
                    f"{field_name}={value!r} is invalid. "
                    f"Allowed values: {sorted(_VALID_8BIT_TYPES)}."
                )


class _NumpyCalibrationReader(CalibrationDataReader):
    """Adapts a numpy calibration array to ORT's CalibrationDataReader API.

    ORT calls get_next() repeatedly and expects either a dict mapping
    input name to numpy array, or None when the data is exhausted.
    """

    def __init__(self, data: np.ndarray, input_name: str) -> None:
        # Each leading-dim slice is one calibration sample; data[i:i+1]
        # preserves the batch dim so the array shape matches the model's
        # expected input (assumed batch=1 in v1).
        self._batches = iter([{input_name: data[i : i + 1]} for i in range(len(data))])

    def get_next(self) -> dict | None:
        return next(self._batches, None)


def quantize(
    fp32_model: onnx.ModelProto,
    calibration_data: np.ndarray,
    input_name: str,
    config: QuantizationConfig | None = None,
) -> onnx.ModelProto:
    """Statically quantize an ONNX model and return the quantized ModelProto.

    The original ModelProto is never mutated. Quantization happens via
    a temp directory that is cleaned up on exit; the returned ModelProto
    is freshly loaded from the quantized output.

    Args:
        fp32_model: Source FP32 model.
        calibration_data: Array of shape (N, *input_dims_without_batch).
            The leading dim is the calibration sample count; each sample
            is fed to ORT individually with batch dim 1.
        input_name: Name of the model input the calibration data feeds.
            Must match a name in fp32_model.graph.input.
        config: Quantization configuration. Defaults to per-channel,
            symmetric, uint8 activations, int8 weights.

    Returns:
        Quantized ModelProto with QuantizeLinear / DequantizeLinear nodes
        inserted around quantizable ops (Conv, MatMul, Gemm).

    Raises:
        QuantizationError: If input_name does not match any model input,
            or if ORT's quantize_static fails (e.g. unsupported op type,
            calibration error).
    """
    if config is None:
        config = QuantizationConfig()

    model_input_names = {inp.name for inp in fp32_model.graph.input}
    if input_name not in model_input_names:
        raise QuantizationError(
            f"input_name={input_name!r} is not an input of the model. "
            f"Available inputs: {sorted(model_input_names)}."
        )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        fp32_path = tmp_dir / "fp32.onnx"
        quant_path = tmp_dir / "quant.onnx"

        onnx.save(fp32_model, str(fp32_path))
        reader = _NumpyCalibrationReader(calibration_data, input_name)

        try:
            quantize_static(
                model_input=str(fp32_path),
                model_output=str(quant_path),
                calibration_data_reader=reader,
                quant_format=QuantFormat.QDQ,
                per_channel=config.per_channel,
                activation_type=_TYPE_TO_QUANTTYPE[config.activation_type],
                weight_type=_TYPE_TO_QUANTTYPE[config.weight_type],
                extra_options={
                    "WeightSymmetric": config.symmetric,
                    "ActivationSymmetric": config.symmetric,
                },
            )
        except Exception as exc:
            raise QuantizationError(f"ORT quantize_static failed: {exc}") from exc

        return onnx.load(str(quant_path))
