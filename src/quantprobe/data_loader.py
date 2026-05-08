"""File I/O for ONNX models and calibration data.

Two pure-loader functions used by the CLI to translate paths into the
in-memory objects the rest of the pipeline operates on. Pushing I/O to
this layer keeps analyzer and friends free of filesystem concerns.

Image-directory calibration loading is planned for v2; v1 supports
.npy and .npz only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx

from quantprobe.exceptions import CalibrationError, ModelLoadError


def load_onnx_model(path: Path) -> onnx.ModelProto:
    """Load and validate an ONNX model from disk.

    Runs ``onnx.checker.check_model`` after loading so semantically broken
    graphs (load fine but cannot run) are caught here rather than deep
    inside ORT later.

    Args:
        path: Filesystem path to a .onnx file.

    Returns:
        Parsed ModelProto.

    Raises:
        ModelLoadError: If the path does not exist, the file cannot be
            parsed as ONNX, or the parsed graph fails validation.
    """
    if not path.exists():
        raise ModelLoadError(f"ONNX model file not found: {path}")

    try:
        model = onnx.load(str(path))
    except Exception as exc:
        raise ModelLoadError(f"Could not parse {path} as ONNX: {exc}") from exc

    try:
        onnx.checker.check_model(model)
    except Exception as exc:
        raise ModelLoadError(f"Model at {path} failed validation: {exc}") from exc

    return model


def load_calibration_data(path: Path) -> np.ndarray:
    """Load a numpy calibration array from a .npy or .npz file.

    Both formats are auto-detected via ``np.load`` (extension is not
    consulted). For .npz archives, the file must contain exactly one
    array -- multi-array archives are rejected because we cannot guess
    which one is the calibration data.

    Args:
        path: Filesystem path to a .npy or .npz file.

    Returns:
        Loaded numpy array.

    Raises:
        CalibrationError: If the path is a directory, does not exist,
            cannot be loaded as numpy data, or is an .npz archive with
            more than one array.
    """
    if path.is_dir():
        raise CalibrationError(
            f"{path} is a directory. Loading calibration data from image "
            "directories is planned for v2; for now, preprocess your images "
            "into a numpy array and save with np.save()."
        )

    if not path.exists():
        raise CalibrationError(f"Calibration data file not found: {path}")

    try:
        loaded = np.load(str(path))
    except Exception as exc:
        raise CalibrationError(f"Could not load {path} as numpy data: {exc}") from exc

    # np.load returns ndarray for .npy, NpzFile for .npz archives.
    if isinstance(loaded, np.ndarray):
        return loaded

    keys = list(loaded.keys())
    if len(keys) != 1:
        loaded.close()
        raise CalibrationError(
            f"{path} contains {len(keys)} arrays; expected exactly one. "
            "Use np.save() to write a single calibration array, or extract "
            "the array you want with np.load(path)['name'] and re-save it."
        )

    array = loaded[keys[0]]
    loaded.close()
    return array
