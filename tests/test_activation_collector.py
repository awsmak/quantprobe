"""Tests for :mod:`quantprobe.activation_collector`.

Run a single class with::

    pytest tests/test_activation_collector.py::TestGraphModification -v
"""

from __future__ import annotations

import numpy as np
import onnx
import pytest

from quantprobe.activation_collector import PASSTHROUGH_OPS, collect
from quantprobe.exceptions import SensitivityError


class TestGraphModification:
    """collect() must expose intermediate tensors without touching the original."""

    def test_original_model_is_not_modified(
        self, two_node_model: onnx.ModelProto, relu_input: np.ndarray
    ) -> None:
        original_output_count = len(two_node_model.graph.output)
        collect(two_node_model, {"input": relu_input})
        # collect works on a deepcopy -- original graph must be untouched
        assert len(two_node_model.graph.output) == original_output_count

    def test_intermediate_tensor_exposed(
        self, two_node_model: onnx.ModelProto, relu_input: np.ndarray
    ) -> None:
        # two_node_model has Relu -> Sigmoid. "hidden" is the Relu output
        # and is not a graph output by default. collect must expose it.
        result = collect(two_node_model, {"input": relu_input})
        assert "hidden" in result

    def test_final_output_still_present(
        self, two_node_model: onnx.ModelProto, relu_input: np.ndarray
    ) -> None:
        result = collect(two_node_model, {"input": relu_input})
        assert "output" in result

    def test_single_node_model_returns_output(
        self, relu_model: onnx.ModelProto, relu_input: np.ndarray
    ) -> None:
        # Relu is not a passthrough op -- its output must appear in result.
        result = collect(relu_model, {"input": relu_input})
        assert "output" in result


class TestPassthroughFiltering:
    """Nodes in PASSTHROUGH_OPS must not be added as extra outputs."""

    def test_passthrough_ops_set_is_not_empty(self) -> None:
        # Sanity check that the constant is defined and populated.
        assert len(PASSTHROUGH_OPS) > 0

    def test_known_ops_are_in_passthrough(self) -> None:
        for op in ("Constant", "Reshape", "Flatten", "Identity", "Transpose"):
            assert op in PASSTHROUGH_OPS

    def test_compute_ops_are_not_in_passthrough(self) -> None:
        for op in ("Conv", "Gemm", "MatMul", "Relu", "Sigmoid", "BatchNormalization"):
            assert op not in PASSTHROUGH_OPS


class TestOutputValues:
    """Collected activation values must be numerically correct."""

    def test_result_values_are_ndarrays(
        self, two_node_model: onnx.ModelProto, relu_input: np.ndarray
    ) -> None:
        result = collect(two_node_model, {"input": relu_input})
        assert all(isinstance(v, np.ndarray) for v in result.values())

    def test_relu_activation_correct(
        self, two_node_model: onnx.ModelProto, relu_input: np.ndarray
    ) -> None:
        # relu_input is [1, -2, 3, -4]. Relu zeroes negatives -> [1, 0, 3, 0].
        result = collect(two_node_model, {"input": relu_input})
        expected = np.array([[1.0, 0.0, 3.0, 0.0]], dtype=np.float32)
        np.testing.assert_array_equal(result["hidden"], expected)

    def test_sigmoid_output_correct(
        self, two_node_model: onnx.ModelProto, relu_input: np.ndarray
    ) -> None:
        # Sigmoid of [1, 0, 3, 0] -> [0.731, 0.5, 0.952, 0.5] approximately.
        result = collect(two_node_model, {"input": relu_input})
        relu_out = np.array([[1.0, 0.0, 3.0, 0.0]], dtype=np.float32)
        expected = 1.0 / (1.0 + np.exp(-relu_out))
        np.testing.assert_allclose(result["output"], expected, rtol=1e-5)

    def test_identical_models_produce_identical_activations(
        self, two_node_model: onnx.ModelProto, relu_input: np.ndarray
    ) -> None:
        # Calling collect twice on the same model must give identical results.
        result_a = collect(two_node_model, {"input": relu_input})
        result_b = collect(two_node_model, {"input": relu_input})
        assert result_a.keys() == result_b.keys()
        for name in result_a:
            np.testing.assert_array_equal(result_a[name], result_b[name])


class TestErrorHandling:
    """collect() must raise SensitivityError for bad inputs."""

    def test_wrong_input_name_raises(
        self, two_node_model: onnx.ModelProto, relu_input: np.ndarray
    ) -> None:
        with pytest.raises(SensitivityError):
            collect(two_node_model, {"wrong_name": relu_input})

    def test_wrong_input_shape_raises(self, two_node_model: onnx.ModelProto) -> None:
        wrong_shape = np.ones((1, 99), dtype=np.float32)
        with pytest.raises(SensitivityError):
            collect(two_node_model, {"input": wrong_shape})
