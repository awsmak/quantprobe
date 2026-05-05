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
import onnx.shape_inference
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


def _make_matmul_model() -> onnx.ModelProto:
    """Build a quantizable single-MatMul model.

    Graph:  input [1, 4] @ weights [4, 3] -> output [1, 3]

    Used by quantizer tests because Relu/Sigmoid alone are not quantized
    by ORT's static quantization -- it only inserts QDQ around Conv,
    MatMul, and Gemm. The weights are baked in as an initializer so the
    quantizer can statically quantize them.
    """
    X = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 4])
    Y = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 3])
    weights = onnx.helper.make_tensor(
        name="weights",
        data_type=onnx.TensorProto.FLOAT,
        dims=[4, 3],
        vals=[
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
            1.1,
            1.2,
        ],
    )
    matmul = onnx.helper.make_node("MatMul", inputs=["input", "weights"], outputs=["output"])
    graph = onnx.helper.make_graph(
        [matmul],
        "matmul_graph",
        inputs=[X],
        outputs=[Y],
        initializer=[weights],
    )
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
def relu_model() -> onnx.ModelProto:
    """Shape-inferred ModelProto for the single-node Relu model.

    Used by activation_collector tests which accept a ModelProto directly
    rather than a path.
    """
    return onnx.shape_inference.infer_shapes(_make_relu_model())


@pytest.fixture
def two_node_model() -> onnx.ModelProto:
    """Shape-inferred ModelProto for the two-node Relu -> Sigmoid model.

    The intermediate tensor 'hidden' is annotated by shape inference but is
    not yet a graph output -- activation_collector must promote it.
    """
    return onnx.shape_inference.infer_shapes(_make_two_node_model())


@pytest.fixture
def relu_input() -> np.ndarray:
    """A [1, 4] float32 input array for the relu/two-node models.

    Mix of positive and negative values so Relu produces a non-trivial output
    (not all zeros, not all pass-through).
    """
    return np.array([[1.0, -2.0, 3.0, -4.0]], dtype=np.float32)


@pytest.fixture
def matmul_model() -> onnx.ModelProto:
    """Shape-inferred ModelProto for the quantizable single-MatMul model."""
    return onnx.shape_inference.infer_shapes(_make_matmul_model())


@pytest.fixture
def matmul_calibration_data() -> np.ndarray:
    """10 deterministic calibration samples for the MatMul model.

    Shape (10, 4) -- the leading dim is the sample count, each row is one
    inference input that gets reshaped to [1, 4] (the model's expected
    input shape) by the calibration reader.
    """
    rng = np.random.default_rng(seed=42)
    return rng.uniform(-1.0, 1.0, size=(10, 4)).astype(np.float32)
