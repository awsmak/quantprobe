"""Tensor-comparison metrics for quantization accuracy analysis.

All functions are pure numpy -- no OnnxRuntime dependency. Inputs are
flattened to 1D and cast to float64 before computation so results are
consistent regardless of input dtype or shape.
"""

from __future__ import annotations

import numpy as np

from quantprobe.exceptions import MetricsError


def _validate_and_prepare(
    reference: np.ndarray, candidate: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Validate shapes, flatten, and cast both arrays to float64."""
    if reference.shape != candidate.shape:
        raise MetricsError(
            f"Shape mismatch: reference {reference.shape} vs candidate {candidate.shape}"
        )
    if reference.size == 0:
        raise MetricsError("Inputs must not be empty")
    return reference.flatten().astype(np.float64), candidate.flatten().astype(np.float64)


def cosine_similarity(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Compute cosine similarity between two tensors.

    Measures the angle between the flattened tensors, independent of magnitude.
    Range is [-1, 1] where 1 means identical direction and -1 means opposite.

    Args:
        reference: FP32 baseline tensor.
        candidate: Quantized tensor to compare against reference.

    Returns:
        Cosine similarity in [-1, 1].

    Raises:
        MetricsError: If shapes mismatch, inputs are empty, or either tensor
            has zero norm.
    """
    ref, cand = _validate_and_prepare(reference, candidate)
    ref_norm = np.linalg.norm(ref)
    cand_norm = np.linalg.norm(cand)
    if ref_norm == 0.0 or cand_norm == 0.0:
        raise MetricsError("Cosine similarity is undefined for zero-norm vectors")
    return float(np.dot(ref, cand) / (ref_norm * cand_norm))


def mse(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Compute mean squared error between two tensors.

    Args:
        reference: FP32 baseline tensor.
        candidate: Quantized tensor to compare against reference.

    Returns:
        Mean squared error (non-negative).

    Raises:
        MetricsError: If shapes mismatch or inputs are empty.
    """
    ref, cand = _validate_and_prepare(reference, candidate)
    return float(np.mean((ref - cand) ** 2))


def max_abs_error(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Compute maximum absolute error between two tensors.

    Catches single-element outliers (e.g. activation clipping) that MSE
    would dilute across the tensor.

    Args:
        reference: FP32 baseline tensor.
        candidate: Quantized tensor to compare against reference.

    Returns:
        Maximum absolute elementwise difference (non-negative).

    Raises:
        MetricsError: If shapes mismatch or inputs are empty.
    """
    ref, cand = _validate_and_prepare(reference, candidate)
    return float(np.max(np.abs(ref - cand)))


def snr_db(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Compute signal-to-noise ratio in decibels.

    Defined as 10 * log10(signal_power / noise_power) where signal_power is
    mean(reference^2) and noise_power is mean((reference - candidate)^2).

    Args:
        reference: FP32 baseline tensor (the signal).
        candidate: Quantized tensor to compare against reference.

    Returns:
        SNR in dB. Returns np.inf when reference and candidate are identical.

    Raises:
        MetricsError: If shapes mismatch, inputs are empty, or the reference
            has zero signal power (SNR undefined).
    """
    ref, cand = _validate_and_prepare(reference, candidate)
    signal_power = float(np.mean(ref**2))
    if signal_power == 0.0:
        raise MetricsError("SNR is undefined when the reference has zero signal power")
    noise_power = float(np.mean((ref - cand) ** 2))
    if noise_power == 0.0:
        return np.inf
    return float(10.0 * np.log10(signal_power / noise_power))
