"""Activation collection via in-memory ONNX graph modification.

Exposes intermediate tensors as graph outputs so ORT returns them during
inference. The original ModelProto is never modified -- all surgery happens
on a deepcopy that is serialized to bytes and never written to disk.

Only onnxruntime is imported here; no other module may import it except
model_runner.py and quantizer.py.
"""

from __future__ import annotations

import copy

import numpy as np
import onnx
import onnxruntime as ort

from quantprobe.exceptions import SensitivityError

# Nodes whose outputs carry no quantization-relevant information.
# Reshape/Transpose/etc. do no arithmetic -- quantization cannot affect them.
# Collecting them would add noise to the sensitivity report without insight.
PASSTHROUGH_OPS: frozenset[str] = frozenset(
    {
        "Cast",
        "Constant",
        "ConstantOfShape",
        "Flatten",
        "Gather",
        "Identity",
        "Reshape",
        "Shape",
        "Squeeze",
        "Transpose",
        "Unsqueeze",
    }
)


def collect(
    model: onnx.ModelProto,
    inputs: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Run inference and return activations for all non-passthrough nodes.

    The model graph is modified in memory to expose intermediate tensors as
    outputs. The caller's ModelProto is never mutated.

    Args:
        model: Shape-inferred ONNX model. Caller must run
            ``onnx.shape_inference.infer_shapes`` before passing it in so
            that intermediate tensor types are annotated in
            ``graph.value_info``.
        inputs: Mapping of input name to numpy array, identical in structure
            to what ``ModelRunner.run`` expects.

    Returns:
        Mapping of tensor name to numpy array for every non-passthrough node
        output, including the model's original declared outputs.

    Raises:
        SensitivityError: If ORT cannot build a session from the modified
            graph, or if inference fails (e.g. wrong input names or shapes).
    """
    modified = _build_modified_model(model)
    return _run_modified_model(modified, inputs)


def _build_modified_model(model: onnx.ModelProto) -> onnx.ModelProto:
    """Return a deepcopy of model with intermediate tensors promoted to outputs."""
    modified = copy.deepcopy(model)

    existing_output_names = {out.name for out in modified.graph.output}

    # value_info holds type/shape annotations for intermediate tensors after
    # shape inference. Build a lookup so we can promote them to outputs.
    value_info_by_name = {vi.name: vi for vi in modified.graph.value_info}

    for node in modified.graph.node:
        if node.op_type in PASSTHROUGH_OPS:
            continue
        for tensor_name in node.output:
            if not tensor_name:
                # Some nodes declare optional outputs with empty string names.
                continue
            if tensor_name in existing_output_names:
                # Already a declared output -- adding it again would make ORT
                # raise a duplicate output error.
                continue
            if tensor_name in value_info_by_name:
                modified.graph.output.append(value_info_by_name[tensor_name])

    return modified


def _run_modified_model(
    model: onnx.ModelProto,
    inputs: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Create an in-memory ORT session and run inference."""
    try:
        opts = ort.SessionOptions()
        opts.log_severity_level = 3  # suppress ORT console noise
        session = ort.InferenceSession(model.SerializeToString(), sess_options=opts)
    except Exception as exc:
        raise SensitivityError(f"Could not build ORT session from modified graph: {exc}") from exc

    output_names = [out.name for out in session.get_outputs()]

    try:
        results = session.run(output_names, inputs)
    except Exception as exc:
        raise SensitivityError(f"Inference failed on modified graph: {exc}") from exc

    return dict(zip(output_names, results))
