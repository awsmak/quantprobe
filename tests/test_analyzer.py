"""Tests for :mod:`quantprobe.analyzer`.

Most tests mock quantizer.quantize and sensitivity.analyse to keep the
orchestration logic isolated from the (slow) real ORT pipeline. One
end-to-end test exercises the whole stack on the matmul fixture to
catch wiring regressions.

Run a single class with::

    pytest tests/test_analyzer.py::TestAnalyseOrchestration -v
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime
from unittest.mock import patch

import numpy as np
import onnx
import pytest

from quantprobe.analyzer import AnalysisResult, analyse
from quantprobe.quantizer import QuantizationConfig
from quantprobe.sensitivity import LayerSensitivity


class TestAnalysisResult:
    """Public dataclass contract."""

    def test_construct_with_all_fields(self) -> None:
        ts = datetime(2026, 5, 4, 12, 0, 0)
        config = QuantizationConfig()
        result = AnalysisResult(
            layers=[],
            config=config,
            calibration_sample_count=10,
            fp32_model_name="test_model",
            timestamp=ts,
        )
        assert result.layers == []
        assert result.config == config
        assert result.calibration_sample_count == 10
        assert result.fp32_model_name == "test_model"
        assert result.timestamp == ts

    def test_is_immutable(self) -> None:
        result = AnalysisResult(
            layers=[],
            config=QuantizationConfig(),
            calibration_sample_count=10,
            fp32_model_name="x",
            timestamp=datetime.now(),
        )
        with pytest.raises(FrozenInstanceError):
            result.fp32_model_name = "y"  # type: ignore[misc]


class TestAnalyseOrchestration:
    """Verify analyse() wires the pipeline correctly using mocks."""

    def test_returns_analysis_result(
        self, matmul_model: onnx.ModelProto, matmul_calibration_data: np.ndarray
    ) -> None:
        with patch("quantprobe.analyzer.quantize", return_value=matmul_model), patch(
            "quantprobe.analyzer.run_sensitivity", return_value=[]
        ):
            result = analyse(matmul_model, matmul_calibration_data, "input")
        assert isinstance(result, AnalysisResult)

    def test_quantize_called_with_provided_config(
        self, matmul_model: onnx.ModelProto, matmul_calibration_data: np.ndarray
    ) -> None:
        config = QuantizationConfig(per_channel=False)
        with patch(
            "quantprobe.analyzer.quantize", return_value=matmul_model
        ) as mock_quantize, patch(
            "quantprobe.analyzer.run_sensitivity", return_value=[]
        ):
            analyse(matmul_model, matmul_calibration_data, "input", config)
        # quantize must have been called once with the explicit config.
        mock_quantize.assert_called_once()
        passed_config = mock_quantize.call_args.kwargs.get("config")
        if passed_config is None:
            # config could also have been passed positionally
            passed_config = mock_quantize.call_args.args[-1]
        assert passed_config == config

    def test_default_config_used_when_none(
        self, matmul_model: onnx.ModelProto, matmul_calibration_data: np.ndarray
    ) -> None:
        with patch("quantprobe.analyzer.quantize", return_value=matmul_model), patch(
            "quantprobe.analyzer.run_sensitivity", return_value=[]
        ):
            result = analyse(matmul_model, matmul_calibration_data, "input")
        assert result.config == QuantizationConfig()

    def test_result_layers_come_from_sensitivity(
        self, matmul_model: onnx.ModelProto, matmul_calibration_data: np.ndarray
    ) -> None:
        fake_layers = [
            LayerSensitivity(
                name="layer_a",
                cosine_similarity=0.99,
                mse=0.01,
                max_abs_error=0.05,
                snr_db=20.0,
            )
        ]
        with patch("quantprobe.analyzer.quantize", return_value=matmul_model), patch(
            "quantprobe.analyzer.run_sensitivity", return_value=fake_layers
        ):
            result = analyse(matmul_model, matmul_calibration_data, "input")
        assert result.layers == fake_layers

    def test_result_records_calibration_sample_count(
        self, matmul_model: onnx.ModelProto, matmul_calibration_data: np.ndarray
    ) -> None:
        with patch("quantprobe.analyzer.quantize", return_value=matmul_model), patch(
            "quantprobe.analyzer.run_sensitivity", return_value=[]
        ):
            result = analyse(matmul_model, matmul_calibration_data, "input")
        assert result.calibration_sample_count == len(matmul_calibration_data)

    def test_result_records_model_name(
        self, matmul_model: onnx.ModelProto, matmul_calibration_data: np.ndarray
    ) -> None:
        # The matmul fixture's graph is named "matmul_graph".
        with patch("quantprobe.analyzer.quantize", return_value=matmul_model), patch(
            "quantprobe.analyzer.run_sensitivity", return_value=[]
        ):
            result = analyse(matmul_model, matmul_calibration_data, "input")
        assert result.fp32_model_name == "matmul_graph"

    def test_result_timestamp_is_recent(
        self, matmul_model: onnx.ModelProto, matmul_calibration_data: np.ndarray
    ) -> None:
        before = datetime.now()
        with patch("quantprobe.analyzer.quantize", return_value=matmul_model), patch(
            "quantprobe.analyzer.run_sensitivity", return_value=[]
        ):
            result = analyse(matmul_model, matmul_calibration_data, "input")
        after = datetime.now()
        assert before <= result.timestamp <= after


class TestAnalyseEndToEnd:
    """One real run end-to-end on the matmul fixture (slow, catches wiring bugs)."""

    def test_full_pipeline_returns_layers(
        self, matmul_model: onnx.ModelProto, matmul_calibration_data: np.ndarray
    ) -> None:
        result = analyse(matmul_model, matmul_calibration_data, "input")
        # Wiring sanity: result populated, sample count matches, at least one
        # layer scored (the MatMul output is a graph output of both models).
        assert isinstance(result, AnalysisResult)
        assert result.calibration_sample_count == 10
        assert len(result.layers) >= 1
        assert all(isinstance(layer, LayerSensitivity) for layer in result.layers)
