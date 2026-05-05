"""Tests for :mod:`quantprobe.quantizer`.

QuantizationConfig validation tests are pure-Python and fast.
quantize() tests hit ORT's quantize_static and are slower (~1s each)
but still small thanks to the tiny MatMul fixture.

Run a single class with::

    pytest tests/test_quantizer.py::TestQuantizationConfigInt4Rejection -v
"""

from __future__ import annotations

import numpy as np
import onnx
import pytest

from quantprobe.exceptions import QuantizationError
from quantprobe.quantizer import QuantizationConfig, quantize


class TestQuantizationConfigDefaults:
    """Default config should construct cleanly with sane values."""

    def test_defaults_construct_without_error(self) -> None:
        config = QuantizationConfig()
        assert config.per_channel is True
        assert config.symmetric is True
        assert config.activation_type == "uint8"
        assert config.weight_type == "int8"

    def test_all_valid_8bit_combinations_accepted(self) -> None:
        # Cartesian product of valid types -- all four combos must work.
        for at in ("int8", "uint8"):
            for wt in ("int8", "uint8"):
                config = QuantizationConfig(activation_type=at, weight_type=wt)
                assert config.activation_type == at
                assert config.weight_type == wt

    def test_per_channel_and_symmetric_toggleable(self) -> None:
        config = QuantizationConfig(per_channel=False, symmetric=False)
        assert config.per_channel is False
        assert config.symmetric is False


class TestQuantizationConfigInt4Rejection:
    """INT4/UINT4 must raise with the helpful pointer message, not a generic error."""

    @pytest.mark.parametrize("value", ["int4", "uint4"])
    def test_int4_in_activation_type_raises_helpful_message(self, value: str) -> None:
        with pytest.raises(QuantizationError) as exc_info:
            QuantizationConfig(activation_type=value)
        msg = str(exc_info.value).lower()
        # Must mention the rejected type and at least one of the alternatives.
        assert value in msg
        assert "matmul4bits" in msg or "vendor" in msg or "v2" in msg

    @pytest.mark.parametrize("value", ["int4", "uint4"])
    def test_int4_in_weight_type_raises_helpful_message(self, value: str) -> None:
        with pytest.raises(QuantizationError) as exc_info:
            QuantizationConfig(weight_type=value)
        msg = str(exc_info.value).lower()
        assert value in msg
        assert "matmul4bits" in msg or "vendor" in msg or "v2" in msg


class TestQuantizationConfigTypoRejection:
    """Unknown values must raise listing the allowed values."""

    @pytest.mark.parametrize("bad_value", ["int7", "uint12", "fp16", "qint8", ""])
    def test_typo_in_activation_type_lists_allowed_values(self, bad_value: str) -> None:
        with pytest.raises(QuantizationError) as exc_info:
            QuantizationConfig(activation_type=bad_value)
        msg = str(exc_info.value).lower()
        # Error must list the legal options so the user can fix it.
        assert "int8" in msg
        assert "uint8" in msg

    @pytest.mark.parametrize("bad_value", ["int7", "uint12", "fp16"])
    def test_typo_in_weight_type_lists_allowed_values(self, bad_value: str) -> None:
        with pytest.raises(QuantizationError) as exc_info:
            QuantizationConfig(weight_type=bad_value)
        msg = str(exc_info.value).lower()
        assert "int8" in msg
        assert "uint8" in msg


class TestQuantize:
    """End-to-end tests for the quantize function. These exercise ORT."""

    def test_returns_model_proto(
        self, matmul_model: onnx.ModelProto, matmul_calibration_data: np.ndarray
    ) -> None:
        result = quantize(matmul_model, matmul_calibration_data, "input")
        assert isinstance(result, onnx.ModelProto)

    def test_original_model_not_modified(
        self, matmul_model: onnx.ModelProto, matmul_calibration_data: np.ndarray
    ) -> None:
        # Same defensive contract as activation_collector -- caller's
        # ModelProto must be untouched.
        original_node_count = len(matmul_model.graph.node)
        quantize(matmul_model, matmul_calibration_data, "input")
        assert len(matmul_model.graph.node) == original_node_count

    def test_quantized_model_contains_qdq_nodes(
        self, matmul_model: onnx.ModelProto, matmul_calibration_data: np.ndarray
    ) -> None:
        # Static QDQ quantization inserts QuantizeLinear and DequantizeLinear
        # nodes around quantized ops. If neither appears, no quantization
        # happened and something is wrong.
        result = quantize(matmul_model, matmul_calibration_data, "input")
        op_types = {node.op_type for node in result.graph.node}
        assert "QuantizeLinear" in op_types
        assert "DequantizeLinear" in op_types

    def test_wrong_input_name_raises(
        self, matmul_model: onnx.ModelProto, matmul_calibration_data: np.ndarray
    ) -> None:
        # Model expects "input"; pass "wrong_name" -- should fail with our
        # error type, not a raw ORT exception.
        with pytest.raises(QuantizationError):
            quantize(matmul_model, matmul_calibration_data, "wrong_name")

    def test_accepts_custom_config(
        self, matmul_model: onnx.ModelProto, matmul_calibration_data: np.ndarray
    ) -> None:
        # Non-default config must be accepted and not break the pipeline.
        config = QuantizationConfig(per_channel=False, symmetric=False)
        result = quantize(matmul_model, matmul_calibration_data, "input", config)
        assert isinstance(result, onnx.ModelProto)
