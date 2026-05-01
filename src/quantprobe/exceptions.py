"""Exception hierarchy for quantprobe.

All quantprobe-specific errors inherit from :class:`QuantProbeError` so that
callers (most importantly the CLI) can distinguish expected, user-facing
failures from unexpected bugs and render them appropriately.
"""

from __future__ import annotations


class QuantProbeError(Exception):
    """Base class for all quantprobe errors.

    The CLI catches this to render a friendly message; anything that does not
    inherit from it is treated as an unexpected bug and shown with a full
    traceback only under ``--debug``.
    """


class ModelLoadError(QuantProbeError):
    """Raised when an ONNX model cannot be loaded, parsed, or validated."""


class CalibrationError(QuantProbeError):
    """Raised when calibration data is missing, malformed, or insufficient."""


class MetricsError(QuantProbeError):
    """Raised when a metric cannot be computed.

    Typical causes: shape mismatch between reference and candidate tensors,
    empty inputs, or numerically degenerate inputs (e.g. a zero-norm vector
    passed to cosine similarity).
    """


class QuantizationError(QuantProbeError):
    """Raised when ORT's quantization API fails or produces an invalid model."""


class SensitivityError(QuantProbeError):
    """Raised when sensitivity analysis cannot be performed.

    Examples: no quantizable layers found, activation collection failed for a
    required tensor, or reference and candidate models have incompatible graphs.
    """


class ReportError(QuantProbeError):
    """Raised when the HTML report cannot be rendered or written."""
