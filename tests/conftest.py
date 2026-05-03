"""Shared pytest fixtures for quantprobe tests.

Fixtures defined here are automatically available to every test file --
no import required. Just add the fixture name as a test parameter.

ONNX models are built programmatically so no binary model files are
committed to the repository.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnx.helper
import pytest


def _make_relu_model() -> onnx.ModelProto:
    """Build a minimal single-node Relu model.

    Graph:  input [1, 4] -> Relu -> output [1, 4]

    Shape [1, 4] is deliberately small so tests run in microseconds.
    Four features gives enough elements to exercise metrics meaningfully.
    """
    X = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 4])
    Y = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 4])
    node = onnx.helper.make_node("Relu", inputs=["input"], outputs=["output"])
    graph = onnx.helper.make_graph([node], "relu_graph", inputs=[X], outputs=[Y])
    model = onnx.helper.make_model(graph)
    model.ir_version = 9
    model.opset_import[0].version = 17
    onnx.checker.check_model(model)
    return model


def _make_two_node_model() -> onnx.ModelProto:
    """Build a two-node model: Relu -> Sigmoid.

    Graph:  input [1, 4] -> Relu -> hidden [1, 4] -> Sigmoid -> output [1, 4]

    Used to test that activation_collector can expose intermediate tensors
    (the hidden tensor between the two nodes is not a graph output by default).
    """
    X = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 4])
    Y = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 4])
    relu_node = onnx.helper.make_node("Relu", inputs=["input"], outputs=["hidden"])
    sigmoid_node = onnx.helper.make_node("Sigmoid", inputs=["hidden"], outputs=["output"])
    graph = onnx.helper.make_graph(
        [relu_node, sigmoid_node], "two_node_graph", inputs=[X], outputs=[Y]
    )
    model = onnx.helper.make_model(graph)
    model.ir_version = 9
    model.opset_import[0].version = 17
    onnx.checker.check_model(model)
    return model


@pytest.fixture
def relu_model_path(tmp_path: Path) -> Path:
    """Path to a saved single-node Relu ONNX model.

    The file lives in pytest's tmp_path directory and is deleted after
    the test session ends.
    """
    path = tmp_path / "relu.onnx"
    onnx.save(_make_relu_model(), str(path))
    return path


@pytest.fixture
def two_node_model_path(tmp_path: Path) -> Path:
    """Path to a saved two-node Relu -> Sigmoid ONNX model."""
    path = tmp_path / "two_node.onnx"
    onnx.save(_make_two_node_model(), str(path))
    return path


@pytest.fixture
def relu_input() -> np.ndarray:
    """A [1, 4] float32 input array for the relu/two-node models.

    Mix of positive and negative values so Relu produces a non-trivial output
    (not all zeros, not all pass-through).
    """
    return np.array([[1.0, -2.0, 3.0, -4.0]], dtype=np.float32)
