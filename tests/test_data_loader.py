"""Tests for :mod:`quantprobe.data_loader`.

Most fixtures here are inline (``tmp_path`` + a quick ``np.save``) because
data_loader is the only consumer of these specific files. Adding them to
conftest would just be noise.

Run a single class with::

    pytest tests/test_data_loader.py::TestLoadCalibrationDataErrors -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import pytest

from quantprobe.data_loader import load_calibration_data, load_onnx_model
from quantprobe.exceptions import CalibrationError, ModelLoadError


class TestLoadOnnxModel:
    """load_onnx_model translates filesystem and parser failures to ModelLoadError."""

    def test_loads_valid_model(self, relu_model_path: Path) -> None:
        model = load_onnx_model(relu_model_path)
        assert isinstance(model, onnx.ModelProto)

    def test_returned_model_has_expected_graph_name(self, relu_model_path: Path) -> None:
        # The relu fixture builds its graph with name "relu_graph". This
        # confirms we got the actual loaded model, not an empty placeholder.
        model = load_onnx_model(relu_model_path)
        assert model.graph.name == "relu_graph"

    def test_nonexistent_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ModelLoadError):
            load_onnx_model(tmp_path / "missing.onnx")

    def test_corrupt_file_raises(self, tmp_path: Path) -> None:
        corrupt = tmp_path / "corrupt.onnx"
        corrupt.write_bytes(b"not a valid onnx model")
        with pytest.raises(ModelLoadError):
            load_onnx_model(corrupt)

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.onnx"
        empty.write_bytes(b"")
        with pytest.raises(ModelLoadError):
            load_onnx_model(empty)


class TestLoadCalibrationDataNpy:
    """The .npy path: single-array files load directly."""

    def test_loads_npy_file(self, tmp_path: Path) -> None:
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        path = tmp_path / "data.npy"
        np.save(path, data)
        loaded = load_calibration_data(path)
        np.testing.assert_array_equal(loaded, data)

    def test_preserves_dtype(self, tmp_path: Path) -> None:
        data = np.array([1, 2, 3], dtype=np.int32)
        path = tmp_path / "data.npy"
        np.save(path, data)
        loaded = load_calibration_data(path)
        assert loaded.dtype == np.int32

    def test_preserves_shape(self, tmp_path: Path) -> None:
        # NCHW vision-model calibration shape.
        data = np.zeros((10, 3, 224, 224), dtype=np.float32)
        path = tmp_path / "data.npy"
        np.save(path, data)
        loaded = load_calibration_data(path)
        assert loaded.shape == (10, 3, 224, 224)


class TestLoadCalibrationDataNpz:
    """The .npz path: archives must contain exactly one array."""

    def test_loads_single_array_npz(self, tmp_path: Path) -> None:
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        path = tmp_path / "data.npz"
        np.savez(path, data)
        loaded = load_calibration_data(path)
        np.testing.assert_array_equal(loaded, data)

    def test_multi_array_npz_raises(self, tmp_path: Path) -> None:
        # Without a way to know which array is the calibration data,
        # the loader refuses ambiguous archives.
        path = tmp_path / "multi.npz"
        np.savez(path, a=np.array([1, 2]), b=np.array([3, 4]))
        with pytest.raises(CalibrationError):
            load_calibration_data(path)


class TestLoadCalibrationDataErrors:
    """Failures translate to CalibrationError with actionable messages."""

    def test_nonexistent_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(CalibrationError):
            load_calibration_data(tmp_path / "missing.npy")

    def test_corrupt_content_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "corrupt.npy"
        bad.write_bytes(b"not a real npy file")
        with pytest.raises(CalibrationError):
            load_calibration_data(bad)

    def test_unknown_extension_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "data.txt"
        bad.write_text("hello")
        with pytest.raises(CalibrationError):
            load_calibration_data(bad)

    def test_directory_path_raises_with_v2_pointer(self, tmp_path: Path) -> None:
        # Image-directory loading is staged for v2; the error must say so
        # so the user knows it's intentional, not a bug.
        with pytest.raises(CalibrationError) as exc_info:
            load_calibration_data(tmp_path)
        msg = str(exc_info.value).lower()
        assert "directory" in msg
        assert "v2" in msg or "np.save" in msg
