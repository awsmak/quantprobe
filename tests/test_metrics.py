"""Tests for :mod:`quantprobe.metrics`.

Each metric has its own ``TestClass`` so a single metric's tests can be run
in isolation while iterating, e.g.::

    pytest tests/test_metrics.py::TestMSE -v
"""

from __future__ import annotations

import numpy as np
import pytest

from quantprobe.exceptions import MetricsError
from quantprobe.metrics import (
    cosine_similarity,
    max_abs_error,
    mse,
    snr_db,
)


class TestSharedValidation:
    """Boundary checks the validation helper must enforce.

    Hosted on ``mse`` for convenience; the same checks should apply to every
    metric (ideally via a shared helper rather than duplicated code).
    """

    def test_shape_mismatch_raises(self) -> None:
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        with pytest.raises(MetricsError):
            mse(a, b)

    def test_empty_inputs_raise(self) -> None:
        a = np.array([], dtype=np.float32)
        b = np.array([], dtype=np.float32)
        with pytest.raises(MetricsError):
            mse(a, b)

    def test_multidimensional_input_flattened(self) -> None:
        # Real activations are N-D (NCHW). An identical 4D tensor -> 0 MSE.
        a = np.arange(8, dtype=np.float32).reshape(1, 2, 2, 2)
        assert mse(a, a) == pytest.approx(0.0)

    def test_mixed_dtypes_match_uniform_dtype(self) -> None:
        a32 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b32 = np.array([2.0, 3.0, 4.0], dtype=np.float32)
        b64 = b32.astype(np.float64)
        assert mse(a32, b64) == pytest.approx(mse(a32, b32))


class TestCosineSimilarity:
    def test_identical_vectors_return_one(self) -> None:
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert cosine_similarity(a, a) == pytest.approx(1.0)

    def test_anti_parallel_vectors_return_minus_one(self) -> None:
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert cosine_similarity(a, -a) == pytest.approx(-1.0)

    def test_orthogonal_vectors_return_zero(self) -> None:
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_scale_invariance(self) -> None:
        # The defining property of cosine: angle, not magnitude. If this fails
        # the function has degenerated into a plain dot product.
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = a * 2.0
        assert cosine_similarity(a, b) == pytest.approx(1.0)

    @pytest.mark.parametrize("zero_position", ["reference", "candidate"])
    def test_zero_norm_input_raises(self, zero_position: str) -> None:
        # 0 / 0 would silently return NaN; both directions are degenerate.
        nonzero = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        zero = np.zeros(3, dtype=np.float32)
        a, b = (zero, nonzero) if zero_position == "reference" else (nonzero, zero)
        with pytest.raises(MetricsError):
            cosine_similarity(a, b)


class TestMSE:
    def test_identical_returns_zero(self) -> None:
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert mse(a, a) == pytest.approx(0.0)

    def test_known_value(self) -> None:
        # All elementwise differences are 1.0 -> mean of squared diffs = 1.0.
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([2.0, 3.0, 4.0], dtype=np.float32)
        assert mse(a, b) == pytest.approx(1.0)

    def test_symmetry(self) -> None:
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 0.0, -1.0], dtype=np.float32)
        assert mse(a, b) == pytest.approx(mse(b, a))

    def test_sign_independence(self) -> None:
        # mean((1 - -1)^2 + (1 - -1)^2) = mean(4 + 4) = 4.
        a = np.array([1.0, 1.0], dtype=np.float32)
        b = np.array([-1.0, -1.0], dtype=np.float32)
        assert mse(a, b) == pytest.approx(4.0)


class TestMaxAbsError:
    def test_identical_returns_zero(self) -> None:
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert max_abs_error(a, a) == pytest.approx(0.0)

    def test_known_value(self) -> None:
        # max(|1-1|, |5-1|, |3-3|) = 4. Confirms it's max, not mean.
        a = np.array([1.0, 5.0, 3.0], dtype=np.float32)
        b = np.array([1.0, 1.0, 3.0], dtype=np.float32)
        assert max_abs_error(a, b) == pytest.approx(4.0)

    def test_single_spike_dominates(self) -> None:
        # The reason this metric exists: catch one clipped element that MSE
        # would dilute across the tensor.
        a = np.array([0.0, 0.0, 0.0, 0.0, 10.0], dtype=np.float32)
        b = np.zeros(5, dtype=np.float32)
        assert max_abs_error(a, b) == pytest.approx(10.0)

    def test_negative_difference_uses_abs(self) -> None:
        a = np.array([-1.0, 0.0], dtype=np.float32)
        b = np.array([3.0, 0.0], dtype=np.float32)
        assert max_abs_error(a, b) == pytest.approx(4.0)


class TestSNRdB:
    def test_known_value(self) -> None:
        # signal_power = mean(1^2) = 1; noise = 0.1 everywhere -> noise_power
        # = 0.01; ratio = 100; 10 * log10(100) = 20 dB.
        ref = np.ones(4, dtype=np.float32)
        cand = np.full(4, 0.9, dtype=np.float32)
        assert snr_db(ref, cand) == pytest.approx(20.0)

    def test_identical_inputs_return_inf(self) -> None:
        # Zero noise -> infinite SNR. "Perfect reconstruction" is a meaningful
        # answer, not an error.
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert snr_db(a, a) == np.inf

    def test_zero_signal_reference_raises(self) -> None:
        # SNR is undefined when the reference carries no signal.
        ref = np.zeros(3, dtype=np.float32)
        cand = np.ones(3, dtype=np.float32)
        with pytest.raises(MetricsError):
            snr_db(ref, cand)
