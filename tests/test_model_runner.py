"""Tests for :mod:`quantprobe.model_runner`.

Each TestClass targets one concern: loading, input validation, inference,
and the public properties. Run a single class with::

    pytest tests/test_model_runner.py::TestModelRunnerLoading -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from quantprobe.exceptions import ModelLoadError
from quantprobe.model_runner import ModelRunner


class TestModelRunnerLoading:
    """ModelRunner.__init__: path validation and session creation."""

    def test_loads_valid_model(self, relu_model_path: Path) -> None:
        # Should not raise -- the happy path.
        runner = ModelRunner(relu_model_path)
        assert runner is not None

    def test_nonexistent_path_raises(self, tmp_path: Path) -> None:
        bad_path = tmp_path / "does_not_exist.onnx"
        with pytest.raises(ModelLoadError):
            ModelRunner(bad_path)

    def test_invalid_file_raises(self, tmp_path: Path) -> None:
        # A file that exists but is not a valid ONNX model.
        corrupt = tmp_path / "corrupt.onnx"
        corrupt.write_bytes(b"this is not an onnx model")
        with pytest.raises(ModelLoadError):
            ModelRunner(corrupt)


class TestModelRunnerProperties:
    """input_names and output_names reflect the loaded model's graph."""

    def test_input_names(self, relu_model_path: Path) -> None:
        runner = ModelRunner(relu_model_path)
        assert runner.input_names == ["input"]

    def test_output_names(self, relu_model_path: Path) -> None:
        runner = ModelRunner(relu_model_path)
        assert runner.output_names == ["output"]

    def test_two_node_model_names(self, two_node_model_path: Path) -> None:
        # Intermediate tensor "hidden" is not an output -- only "output" is.
        runner = ModelRunner(two_node_model_path)
        assert runner.input_names == ["input"]
        assert runner.output_names == ["output"]


class TestModelRunnerInference:
    """ModelRunner.run: input validation and output correctness."""

    def test_run_returns_dict(self, relu_model_path: Path, relu_input: np.ndarray) -> None:
        runner = ModelRunner(relu_model_path)
        result = runner.run({"input": relu_input})
        assert isinstance(result, dict)

    def test_run_output_names_match_model(
        self, relu_model_path: Path, relu_input: np.ndarray
    ) -> None:
        runner = ModelRunner(relu_model_path)
        result = runner.run({"input": relu_input})
        assert list(result.keys()) == ["output"]

    def test_run_output_values_are_ndarrays(
        self, relu_model_path: Path, relu_input: np.ndarray
    ) -> None:
        runner = ModelRunner(relu_model_path)
        result = runner.run({"input": relu_input})
        assert all(isinstance(v, np.ndarray) for v in result.values())

    def test_relu_zeroes_negatives(self, relu_model_path: Path, relu_input: np.ndarray) -> None:
        # relu_input is [1, -2, 3, -4]. Relu sets negatives to zero.
        # Expected: [1, 0, 3, 0].
        runner = ModelRunner(relu_model_path)
        result = runner.run({"input": relu_input})
        expected = np.array([[1.0, 0.0, 3.0, 0.0]], dtype=np.float32)
        np.testing.assert_array_equal(result["output"], expected)

    def test_wrong_input_name_raises(self, relu_model_path: Path, relu_input: np.ndarray) -> None:
        # Model expects "input", we pass "images" -- should fail with our error,
        # not a raw ORT exception.
        runner = ModelRunner(relu_model_path)
        with pytest.raises(ModelLoadError):
            runner.run({"images": relu_input})

    def test_wrong_input_shape_raises(self, relu_model_path: Path) -> None:
        # Model expects [1, 4], we pass [1, 8].
        runner = ModelRunner(relu_model_path)
        wrong_shape = np.ones((1, 8), dtype=np.float32)
        with pytest.raises(ModelLoadError):
            runner.run({"input": wrong_shape})
