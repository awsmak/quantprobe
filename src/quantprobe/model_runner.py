"""ORT inference wrapper for quantprobe.

All onnxruntime-specific logic lives here. No other module may import
onnxruntime directly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

from quantprobe.exceptions import ModelLoadError


class ModelRunner:
    """Loads a single ONNX model and runs CPU inference via OnnxRuntime.

    One instance wraps one model. To compare FP32 vs quantized outputs,
    create two separate runners and call run() on each.

    Args:
        model_path: Path to a valid ONNX model file.

    Raises:
        ModelLoadError: If the path does not exist or ORT cannot load the model.
    """

    def __init__(self, model_path: Path) -> None:
        if not model_path.exists():
            raise ModelLoadError(f"Model file not found: {model_path}")

        try:
            opts = ort.SessionOptions()
            opts.log_severity_level = 3  # suppress ORT's own console output
            self._session = ort.InferenceSession(str(model_path), sess_options=opts)
        except Exception as exc:
            raise ModelLoadError(f"ORT could not load model at {model_path}: {exc}") from exc

        # Cache names and expected shapes once at load time.
        self._input_meta = {inp.name: inp.shape for inp in self._session.get_inputs()}
        self._output_names: list[str] = [out.name for out in self._session.get_outputs()]

    @property
    def input_names(self) -> list[str]:
        """Names of the model's input tensors."""
        return list(self._input_meta.keys())

    @property
    def output_names(self) -> list[str]:
        """Names of the model's output tensors."""
        return list(self._output_names)

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run inference and return all declared outputs by name.

        Args:
            inputs: Mapping of input tensor name to numpy array. Must
                contain exactly the names the model expects.

        Returns:
            Mapping of output tensor name to numpy array.

        Raises:
            ModelLoadError: If input names do not match the model, or if
                input shapes are incompatible with the model's expectations.
        """
        self._validate_inputs(inputs)
        results = self._session.run(self._output_names, inputs)
        return dict(zip(self._output_names, results))

    def _validate_inputs(self, inputs: dict[str, np.ndarray]) -> None:
        """Raise ModelLoadError if input names or shapes do not match."""
        expected = set(self._input_meta.keys())
        provided = set(inputs.keys())

        if expected != provided:
            raise ModelLoadError(
                f"Input name mismatch: model expects {sorted(expected)} but got {sorted(provided)}"
            )

        for name, array in inputs.items():
            expected_shape = self._input_meta[name]
            # ORT shapes may contain None for dynamic dimensions -- only
            # check axes where the model declares a concrete size.
            if expected_shape is not None:
                for _, (got, want) in enumerate(zip(array.shape, expected_shape)):
                    if want is not None and got != want:
                        raise ModelLoadError(
                            f"Input '{name}' has wrong shape: "
                            f"expected {expected_shape}, got {array.shape}"
                        )
