import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class NpuRewriteRule:
    name: str
    description: str
    apply: Callable[[Path, Path], int]


@dataclass(frozen=True)
class AppliedRewriteRule:
    name: str
    description: str
    rewritten_count: int


def default_npu_rewrite_rules() -> tuple[NpuRewriteRule, ...]:
    return (
        NpuRewriteRule(
            "float16_dequantize_constants",
            "Fold constant FLOAT16->FLOAT32 DEQUANTIZE ops offline",
            rewrite_float16_dequantize_constants,
        ),
        NpuRewriteRule(
            "int8_conv_biases_to_int32",
            "Rewrite invalid INT8 Conv/DepthwiseConv bias tensors to INT32",
            rewrite_int8_conv_biases_to_int32,
        ),
        NpuRewriteRule(
            "prelu_float_islands_to_int8",
            "Rewrite DEQUANTIZE->PRELU->QUANTIZE float islands to int8 PReLU",
            rewrite_prelu_float_islands_to_int8,
        ),
        NpuRewriteRule(
            "serving_default_signature",
            "Ensure a serving_default signature exists for quantizer/runtime tooling",
            _ensure_serving_default_signature_count,
        ),
    )


def apply_npu_rewrite_rules(
    input_path: str | Path,
    output_path: str | Path,
    *,
    rules: tuple[NpuRewriteRule, ...] | None = None,
    rule_names: tuple[str, ...] | list[str] | None = None,
) -> list[AppliedRewriteRule]:
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected_rules = rules or default_npu_rewrite_rules()
    if rule_names is not None:
        allowed = set(rule_names)
        selected_rules = tuple(rule for rule in selected_rules if rule.name in allowed)

    current_path = input_path
    applied: list[AppliedRewriteRule] = []
    scratch_paths: list[Path] = []
    for index, rule in enumerate(selected_rules):
        next_path = output_path.with_name(f"{output_path.stem}.{index}.{rule.name}.tmp.tflite")
        rewritten_count = rule.apply(current_path, next_path)
        applied.append(
            AppliedRewriteRule(
                name=rule.name,
                description=rule.description,
                rewritten_count=rewritten_count,
            )
        )
        current_path = next_path
        scratch_paths.append(next_path)

    if selected_rules:
        shutil.copy2(current_path, output_path)
    else:
        shutil.copy2(input_path, output_path)

    for scratch_path in scratch_paths:
        if scratch_path.exists():
            scratch_path.unlink()
    return applied


def _ensure_serving_default_signature_count(input_path: Path, output_path: Path) -> int:
    return 1 if ensure_serving_default_signature(input_path, output_path) else 0



def ensure_serving_default_signature(
    input_path: str | Path, output_path: str | Path
) -> bool:
    """Ensure a TFLite model exposes a serving_default signature.

    ai-edge-quantizer calibrates via TFLite signatures. Some prebuilt MediaPipe
    models have regular subgraph inputs/outputs but no SignatureDef, so add a
    simple signature that maps each tensor name to its subgraph tensor index.
    """
    from ai_edge_litert import schema_py_generated as schema
    from ai_edge_litert.tools import flatbuffer_utils

    model = flatbuffer_utils.read_model(input_path)
    if any(
        signature.signatureKey == b"serving_default"
        for signature in (model.signatureDefs or [])
    ):
        flatbuffer_utils.write_model(model, output_path)
        return False

    if not model.subgraphs:
        raise ValueError("cannot add signature to a TFLite model with no subgraphs")
    subgraph = model.subgraphs[0]

    signature = schema.SignatureDefT()
    signature.signatureKey = b"serving_default"
    signature.subgraphIndex = 0
    signature.inputs = [_tensor_map(schema, subgraph, index) for index in subgraph.inputs]
    signature.outputs = [_tensor_map(schema, subgraph, index) for index in subgraph.outputs]
    model.signatureDefs = [*(model.signatureDefs or []), signature]

    flatbuffer_utils.write_model(model, output_path)
    return True


def _tensor_map(schema, subgraph, tensor_index: int):
    tensor_map = schema.TensorMapT()
    name = subgraph.tensors[tensor_index].name
    tensor_map.name = name if isinstance(name, bytes) else str(name).encode("utf-8")
    tensor_map.tensorIndex = int(tensor_index)
    return tensor_map
def rewrite_float16_dequantize_constants(
    input_path: str | Path, output_path: str | Path
) -> int:
    """Fold constant FLOAT16 -> FLOAT32 DEQUANTIZE ops into model buffers.

    MediaPipe ships some TFLite models with float16 weights followed by runtime
    DEQUANTIZE ops. TFLM then allocates the expanded float32 tensors in SRAM.
    Folding those constants offline trades flash for arena headroom.
    """
    from ai_edge_litert import schema_py_generated as schema
    from ai_edge_litert.tools import flatbuffer_utils

    model = flatbuffer_utils.read_model(input_path)
    rewritten_count = 0

    for subgraph in model.subgraphs:
        replacements: dict[int, int] = {}
        remove_indexes: set[int] = set()

        for op_index, op in enumerate(subgraph.operators):
            opcode = model.operatorCodes[op.opcodeIndex]
            if (
                flatbuffer_utils.get_builtin_code_from_operator_code(opcode)
                != schema.BuiltinOperator.DEQUANTIZE
            ):
                continue
            if len(op.inputs) != 1 or len(op.outputs) != 1:
                continue

            input_index = op.inputs[0]
            output_index = op.outputs[0]
            input_tensor = subgraph.tensors[input_index]
            output_tensor = subgraph.tensors[output_index]
            if (
                input_tensor.type != schema.TensorType.FLOAT16
                or output_tensor.type != schema.TensorType.FLOAT32
            ):
                continue

            buffer = model.buffers[input_tensor.buffer]
            if buffer.data is None or len(buffer.data) == 0:
                continue

            float16_values = np.frombuffer(bytes(buffer.data), dtype=np.float16)
            float32_values = float16_values.astype(np.float32)
            buffer.data = np.frombuffer(float32_values.tobytes(), dtype=np.uint8)
            input_tensor.type = schema.TensorType.FLOAT32

            replacements[output_index] = input_index
            remove_indexes.add(op_index)
            rewritten_count += 1

        if replacements:
            for op_index, op in enumerate(subgraph.operators):
                if op_index in remove_indexes:
                    continue
                op.inputs = [replacements.get(tensor, tensor) for tensor in op.inputs]
            subgraph.inputs = [
                replacements.get(tensor, tensor) for tensor in subgraph.inputs
            ]
            subgraph.outputs = [
                replacements.get(tensor, tensor) for tensor in subgraph.outputs
            ]
            subgraph.operators = [
                op
                for op_index, op in enumerate(subgraph.operators)
                if op_index not in remove_indexes
            ]
            _compact_subgraph_tensors(subgraph)

    flatbuffer_utils.write_model(model, output_path)
    return rewritten_count


def rewrite_prelu_float_islands_to_int8(
    input_path: str | Path, output_path: str | Path
) -> int:
    """Rewrite INT8->FLOAT32 PReLU islands back to quantized PReLU.

    Some MediaPipe models quantize convolutions but leave PReLU activations as
    FLOAT32 islands: DEQUANTIZE -> PRELU -> QUANTIZE. TFLM can run quantized
    PReLU, and Vela can then keep the activation path on the NPU. The alpha
    tensor is quantized per tensor with zero point 0.
    """
    from ai_edge_litert import schema_py_generated as schema
    from ai_edge_litert.tools import flatbuffer_utils

    model = flatbuffer_utils.read_model(input_path)
    rewritten_count = 0

    for subgraph in model.subgraphs:
        remove_indexes: set[int] = set()

        for op_index, op in enumerate(list(subgraph.operators)):
            if not _operator_is(model, op, schema.BuiltinOperator.PRELU):
                continue
            if op_index == 0 or op_index + 1 >= len(subgraph.operators):
                continue

            dequantize_op = subgraph.operators[op_index - 1]
            quantize_op = subgraph.operators[op_index + 1]
            if not _operator_is(model, dequantize_op, schema.BuiltinOperator.DEQUANTIZE):
                continue
            if not _operator_is(model, quantize_op, schema.BuiltinOperator.QUANTIZE):
                continue
            if len(dequantize_op.inputs) != 1 or len(dequantize_op.outputs) != 1:
                continue
            if len(op.inputs) != 2 or len(op.outputs) != 1:
                continue
            if len(quantize_op.inputs) != 1 or len(quantize_op.outputs) != 1:
                continue
            if dequantize_op.outputs[0] != op.inputs[0]:
                continue
            if op.outputs[0] != quantize_op.inputs[0]:
                continue

            input_index = dequantize_op.inputs[0]
            alpha_index = op.inputs[1]
            output_index = quantize_op.outputs[0]
            input_tensor = subgraph.tensors[input_index]
            alpha_tensor = subgraph.tensors[alpha_index]
            output_tensor = subgraph.tensors[output_index]
            if input_tensor.type != schema.TensorType.INT8:
                continue
            if alpha_tensor.type != schema.TensorType.FLOAT32:
                continue
            if output_tensor.type != schema.TensorType.INT8:
                continue

            buffer = model.buffers[alpha_tensor.buffer]
            if buffer.data is None or len(buffer.data) == 0:
                continue

            alpha_values = np.frombuffer(bytes(buffer.data), dtype=np.float32)
            if alpha_values.size == 0:
                continue
            max_abs = float(np.max(np.abs(alpha_values)))
            alpha_scale = max(max_abs / 127.0, 1e-9)
            quantized_alpha = np.clip(
                np.round(alpha_values / alpha_scale), -128, 127
            ).astype(np.int8)

            buffer.data = np.frombuffer(quantized_alpha.tobytes(), dtype=np.uint8)
            alpha_tensor.type = schema.TensorType.INT8
            alpha_tensor.quantization = schema.QuantizationParametersT()
            alpha_tensor.quantization.scale = np.array([alpha_scale], dtype=np.float32)
            alpha_tensor.quantization.zeroPoint = np.array([0], dtype=np.int64)
            alpha_tensor.quantization.quantizedDimension = 0

            op.inputs = [input_index, alpha_index]
            op.outputs = [output_index]
            remove_indexes.update({op_index - 1, op_index + 1})
            rewritten_count += 1

        if remove_indexes:
            subgraph.operators = [
                op
                for op_index, op in enumerate(subgraph.operators)
                if op_index not in remove_indexes
            ]
            _compact_subgraph_tensors(subgraph)

    flatbuffer_utils.write_model(model, output_path)
    return rewritten_count


def rewrite_int8_conv_biases_to_int32(
    input_path: str | Path, output_path: str | Path
) -> int:
    """Convert invalid INT8 Conv/DepthwiseConv bias tensors to INT32.

    Full-integer TFLite Conv2D/DepthwiseConv2D kernels expect bias tensors to be
    int32 with scale = input_scale * filter_scale. Some third-party prebuilt
    MobileFaceNet TFLite assets carry int8 bias tensors after static
    quantization; Vela then leaves those ops on CPU and TFLM can hang at Invoke.
    """
    from ai_edge_litert import schema_py_generated as schema
    from ai_edge_litert.tools import flatbuffer_utils

    model = flatbuffer_utils.read_model(input_path)
    rewritten_count = 0

    for subgraph in model.subgraphs:
        for op in subgraph.operators:
            if not (
                _operator_is(model, op, schema.BuiltinOperator.CONV_2D)
                or _operator_is(model, op, schema.BuiltinOperator.DEPTHWISE_CONV_2D)
            ):
                continue
            if len(op.inputs) < 3 or op.inputs[2] < 0:
                continue

            input_tensor = subgraph.tensors[op.inputs[0]]
            filter_tensor = subgraph.tensors[op.inputs[1]]
            bias_tensor = subgraph.tensors[op.inputs[2]]
            if bias_tensor.type != schema.TensorType.INT8:
                continue

            input_scale = _quant_scales(input_tensor)
            filter_scale = _quant_scales(filter_tensor)
            bias_scale = _quant_scales(bias_tensor)
            bias_zero_point = _quant_zero_points(bias_tensor)
            if input_scale.size != 1 or filter_scale.size == 0 or bias_scale.size == 0:
                continue

            bias_buffer = model.buffers[bias_tensor.buffer]
            if bias_buffer.data is None or len(bias_buffer.data) == 0:
                continue
            int8_values = np.frombuffer(bytes(bias_buffer.data), dtype=np.int8)
            if int8_values.size == 0:
                continue

            target_scale = input_scale.astype(np.float64) * filter_scale.astype(np.float64)
            if target_scale.size == 1 and int8_values.size != 1:
                target_scale = np.repeat(target_scale, int8_values.size)
            if target_scale.size != int8_values.size:
                continue

            if bias_zero_point.size == 1:
                bias_zero_point = np.repeat(bias_zero_point, int8_values.size)
            if bias_scale.size == 1:
                bias_scale = np.repeat(bias_scale, int8_values.size)
            if bias_zero_point.size != int8_values.size or bias_scale.size != int8_values.size:
                continue

            real_bias = (int8_values.astype(np.float64) - bias_zero_point) * bias_scale
            int32_values = np.clip(
                np.round(real_bias / target_scale),
                np.iinfo(np.int32).min,
                np.iinfo(np.int32).max,
            ).astype(np.int32)

            bias_buffer.data = np.frombuffer(int32_values.tobytes(), dtype=np.uint8)
            bias_tensor.type = schema.TensorType.INT32
            bias_tensor.quantization = schema.QuantizationParametersT()
            bias_tensor.quantization.scale = target_scale.astype(np.float32)
            bias_tensor.quantization.zeroPoint = np.zeros(target_scale.size, dtype=np.int64)
            bias_tensor.quantization.quantizedDimension = (
                filter_tensor.quantization.quantizedDimension
                if filter_tensor.quantization is not None
                else 0
            )
            rewritten_count += 1

    flatbuffer_utils.write_model(model, output_path)
    return rewritten_count


def _operator_is(model, operator, builtin_code: int) -> bool:
    from ai_edge_litert.tools import flatbuffer_utils

    opcode = model.operatorCodes[operator.opcodeIndex]
    return flatbuffer_utils.get_builtin_code_from_operator_code(opcode) == builtin_code


def _quant_scales(tensor) -> np.ndarray:
    if tensor.quantization is None or tensor.quantization.scale is None:
        return np.array([], dtype=np.float64)
    return np.asarray(tensor.quantization.scale, dtype=np.float64)


def _quant_zero_points(tensor) -> np.ndarray:
    if tensor.quantization is None or tensor.quantization.zeroPoint is None:
        return np.array([0], dtype=np.int64)
    return np.asarray(tensor.quantization.zeroPoint, dtype=np.int64)

def _compact_subgraph_tensors(subgraph) -> None:
    used_indexes: set[int] = set()

    def add(indexes: list[int] | None) -> None:
        if indexes is None:
            return
        used_indexes.update(index for index in indexes if index >= 0)

    add(subgraph.inputs)
    add(subgraph.outputs)
    for op in subgraph.operators:
        add(op.inputs)
        add(op.outputs)
        add(op.intermediates)

    index_map = {old: new for new, old in enumerate(sorted(used_indexes))}

    def remap(indexes: list[int] | None) -> list[int] | None:
        if indexes is None:
            return None
        return [index_map[index] if index >= 0 else index for index in indexes]

    subgraph.tensors = [subgraph.tensors[index] for index in sorted(used_indexes)]
    subgraph.inputs = remap(subgraph.inputs)
    subgraph.outputs = remap(subgraph.outputs)
    for op in subgraph.operators:
        op.inputs = remap(op.inputs)
        op.outputs = remap(op.outputs)
        op.intermediates = remap(op.intermediates)
