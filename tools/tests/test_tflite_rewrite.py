from pathlib import Path

import numpy as np

from tools.model_converter.tflite_rewrite import (
    ensure_serving_default_signature,
    rewrite_float16_dequantize_constants,
    rewrite_int8_conv_biases_to_int32,
    rewrite_prelu_float_islands_to_int8,
)


def _write_float16_dequantize_model(path: Path) -> None:
    from ai_edge_litert import schema_py_generated as schema
    from ai_edge_litert.tools import flatbuffer_utils

    model = schema.ModelT()
    model.version = 3

    opcode = schema.OperatorCodeT()
    opcode.builtinCode = schema.BuiltinOperator.DEQUANTIZE
    opcode.deprecatedBuiltinCode = schema.BuiltinOperator.DEQUANTIZE
    opcode.version = 1
    model.operatorCodes = [opcode]

    empty_buffer = schema.BufferT()
    weights_buffer = schema.BufferT()
    weights_buffer.data = np.frombuffer(
        np.array([1.5, -2.0], dtype=np.float16).tobytes(), dtype=np.uint8
    )
    model.buffers = [empty_buffer, weights_buffer]

    weights = schema.TensorT()
    weights.name = "weights_f16"
    weights.shape = [2]
    weights.shapeSignature = [2]
    weights.type = schema.TensorType.FLOAT16
    weights.buffer = 1

    dequantized = schema.TensorT()
    dequantized.name = "weights_f32"
    dequantized.shape = [2]
    dequantized.shapeSignature = [2]
    dequantized.type = schema.TensorType.FLOAT32
    dequantized.buffer = 0

    op = schema.OperatorT()
    op.opcodeIndex = 0
    op.inputs = [0]
    op.outputs = [1]

    subgraph = schema.SubGraphT()
    subgraph.name = "main"
    subgraph.tensors = [weights, dequantized]
    subgraph.inputs = []
    subgraph.outputs = [1]
    subgraph.operators = [op]
    model.subgraphs = [subgraph]

    flatbuffer_utils.write_model(model, path)


def test_rewrite_float16_dequantize_constants_removes_runtime_dequantize(
    tmp_path: Path,
) -> None:
    from ai_edge_litert import schema_py_generated as schema
    from ai_edge_litert.tools import flatbuffer_utils

    input_path = tmp_path / "input.tflite"
    output_path = tmp_path / "output.tflite"
    _write_float16_dequantize_model(input_path)

    rewritten = rewrite_float16_dequantize_constants(input_path, output_path)

    model = flatbuffer_utils.read_model(output_path)
    subgraph = model.subgraphs[0]
    assert rewritten == 1
    assert subgraph.operators == []
    assert subgraph.outputs == [0]
    assert len(subgraph.tensors) == 1
    assert subgraph.tensors[0].type == schema.TensorType.FLOAT32

    values = np.frombuffer(bytes(model.buffers[1].data), dtype=np.float32)
    np.testing.assert_allclose(values, np.array([1.5, -2.0], dtype=np.float32))


def test_ensure_serving_default_signature_maps_subgraph_inputs_and_outputs(
    tmp_path: Path,
) -> None:
    from ai_edge_litert.tools import flatbuffer_utils

    input_path = tmp_path / "input.tflite"
    output_path = tmp_path / "output.tflite"
    _write_float_prelu_island_model(input_path)

    added = ensure_serving_default_signature(input_path, output_path)

    model = flatbuffer_utils.read_model(output_path)
    signatures = model.signatureDefs
    assert added is True
    assert signatures is not None
    assert len(signatures) == 1
    assert signatures[0].signatureKey == b"serving_default"
    assert [(m.name, m.tensorIndex) for m in signatures[0].inputs] == [
        (b"input_i8", 0)
    ]
    assert [(m.name, m.tensorIndex) for m in signatures[0].outputs] == [
        (b"output_i8", 4)
    ]


def _write_float_prelu_island_model(path: Path) -> None:
    from ai_edge_litert import schema_py_generated as schema
    from ai_edge_litert.tools import flatbuffer_utils

    model = schema.ModelT()
    model.version = 3

    opcodes = []
    for code in (
        schema.BuiltinOperator.DEQUANTIZE,
        schema.BuiltinOperator.PRELU,
        schema.BuiltinOperator.QUANTIZE,
    ):
        opcode = schema.OperatorCodeT()
        opcode.builtinCode = code
        opcode.deprecatedBuiltinCode = code
        opcode.version = 1
        opcodes.append(opcode)
    model.operatorCodes = opcodes

    empty_buffer = schema.BufferT()
    alpha_buffer = schema.BufferT()
    alpha_buffer.data = np.frombuffer(
        np.array([0.25, -0.5], dtype=np.float32).tobytes(), dtype=np.uint8
    )
    model.buffers = [empty_buffer, alpha_buffer]

    input_tensor = schema.TensorT()
    input_tensor.name = "input_i8"
    input_tensor.shape = [1, 1, 1, 2]
    input_tensor.shapeSignature = [1, 1, 1, 2]
    input_tensor.type = schema.TensorType.INT8
    input_tensor.buffer = 0
    input_tensor.quantization = schema.QuantizationParametersT()
    input_tensor.quantization.scale = np.array([0.1], dtype=np.float32)
    input_tensor.quantization.zeroPoint = np.array([-3], dtype=np.int64)

    dequant_tensor = schema.TensorT()
    dequant_tensor.name = "input_f32"
    dequant_tensor.shape = [1, 1, 1, 2]
    dequant_tensor.shapeSignature = [1, 1, 1, 2]
    dequant_tensor.type = schema.TensorType.FLOAT32
    dequant_tensor.buffer = 0

    alpha_tensor = schema.TensorT()
    alpha_tensor.name = "alpha_f32"
    alpha_tensor.shape = [1, 1, 2]
    alpha_tensor.shapeSignature = [1, 1, 2]
    alpha_tensor.type = schema.TensorType.FLOAT32
    alpha_tensor.buffer = 1

    prelu_tensor = schema.TensorT()
    prelu_tensor.name = "prelu_f32"
    prelu_tensor.shape = [1, 1, 1, 2]
    prelu_tensor.shapeSignature = [1, 1, 1, 2]
    prelu_tensor.type = schema.TensorType.FLOAT32
    prelu_tensor.buffer = 0

    output_tensor = schema.TensorT()
    output_tensor.name = "output_i8"
    output_tensor.shape = [1, 1, 1, 2]
    output_tensor.shapeSignature = [1, 1, 1, 2]
    output_tensor.type = schema.TensorType.INT8
    output_tensor.buffer = 0
    output_tensor.quantization = schema.QuantizationParametersT()
    output_tensor.quantization.scale = np.array([0.2], dtype=np.float32)
    output_tensor.quantization.zeroPoint = np.array([5], dtype=np.int64)

    dequant = schema.OperatorT()
    dequant.opcodeIndex = 0
    dequant.inputs = [0]
    dequant.outputs = [1]

    prelu = schema.OperatorT()
    prelu.opcodeIndex = 1
    prelu.inputs = [1, 2]
    prelu.outputs = [3]

    quantize = schema.OperatorT()
    quantize.opcodeIndex = 2
    quantize.inputs = [3]
    quantize.outputs = [4]

    subgraph = schema.SubGraphT()
    subgraph.name = "main"
    subgraph.tensors = [
        input_tensor,
        dequant_tensor,
        alpha_tensor,
        prelu_tensor,
        output_tensor,
    ]
    subgraph.inputs = [0]
    subgraph.outputs = [4]
    subgraph.operators = [dequant, prelu, quantize]
    model.subgraphs = [subgraph]

    flatbuffer_utils.write_model(model, path)


def test_rewrite_prelu_float_islands_to_int8_removes_dequant_quant_boundaries(
    tmp_path: Path,
) -> None:
    from ai_edge_litert import schema_py_generated as schema
    from ai_edge_litert.tools import flatbuffer_utils

    input_path = tmp_path / "input.tflite"
    output_path = tmp_path / "output.tflite"
    _write_float_prelu_island_model(input_path)

    rewritten = rewrite_prelu_float_islands_to_int8(input_path, output_path)

    model = flatbuffer_utils.read_model(output_path)
    subgraph = model.subgraphs[0]
    op = subgraph.operators[0]
    code = flatbuffer_utils.get_builtin_code_from_operator_code(
        model.operatorCodes[op.opcodeIndex]
    )
    alpha = subgraph.tensors[op.inputs[1]]
    alpha_values = np.frombuffer(
        bytes(model.buffers[alpha.buffer].data), dtype=np.int8
    )

    assert rewritten == 1
    assert len(subgraph.operators) == 1
    assert code == schema.BuiltinOperator.PRELU
    assert subgraph.tensors[op.inputs[0]].type == schema.TensorType.INT8
    assert alpha.type == schema.TensorType.INT8
    assert subgraph.tensors[op.outputs[0]].type == schema.TensorType.INT8
    assert alpha.quantization.zeroPoint.tolist() == [0]
    assert alpha_values.tolist() == [64, -127]


def _write_int8_bias_conv_model(path: Path) -> None:
    from ai_edge_litert import schema_py_generated as schema
    from ai_edge_litert.tools import flatbuffer_utils

    model = schema.ModelT()
    model.version = 3

    opcode = schema.OperatorCodeT()
    opcode.builtinCode = schema.BuiltinOperator.CONV_2D
    opcode.deprecatedBuiltinCode = schema.BuiltinOperator.CONV_2D
    opcode.version = 1
    model.operatorCodes = [opcode]

    empty_buffer = schema.BufferT()
    filter_buffer = schema.BufferT()
    filter_buffer.data = np.frombuffer(np.array([1, -2], dtype=np.int8).tobytes(), dtype=np.uint8)
    bias_buffer = schema.BufferT()
    bias_buffer.data = np.frombuffer(np.array([10, -20], dtype=np.int8).tobytes(), dtype=np.uint8)
    model.buffers = [empty_buffer, filter_buffer, bias_buffer]

    input_tensor = _quant_tensor(schema, "input", [1, 1, 1, 1], schema.TensorType.INT8, 0, [0.25], [0])
    filter_tensor = _quant_tensor(schema, "filter", [2, 1, 1, 1], schema.TensorType.INT8, 1, [0.5, 0.25], [0, 0])
    filter_tensor.quantization.quantizedDimension = 0
    bias_tensor = _quant_tensor(schema, "bias", [2], schema.TensorType.INT8, 2, [0.125, 0.25], [0, 0])
    output_tensor = _quant_tensor(schema, "output", [1, 1, 1, 2], schema.TensorType.INT8, 0, [0.5], [0])

    op = schema.OperatorT()
    op.opcodeIndex = 0
    op.inputs = [0, 1, 2]
    op.outputs = [3]

    subgraph = schema.SubGraphT()
    subgraph.name = "main"
    subgraph.tensors = [input_tensor, filter_tensor, bias_tensor, output_tensor]
    subgraph.inputs = [0]
    subgraph.outputs = [3]
    subgraph.operators = [op]
    model.subgraphs = [subgraph]

    flatbuffer_utils.write_model(model, path)


def _quant_tensor(schema, name, shape, tensor_type, buffer, scale, zero_point):
    tensor = schema.TensorT()
    tensor.name = name
    tensor.shape = shape
    tensor.shapeSignature = shape
    tensor.type = tensor_type
    tensor.buffer = buffer
    tensor.quantization = schema.QuantizationParametersT()
    tensor.quantization.scale = np.array(scale, dtype=np.float32)
    tensor.quantization.zeroPoint = np.array(zero_point, dtype=np.int64)
    tensor.quantization.quantizedDimension = 0
    return tensor


def test_rewrite_int8_conv_biases_to_int32_uses_input_filter_scales(
    tmp_path: Path,
) -> None:
    from ai_edge_litert import schema_py_generated as schema
    from ai_edge_litert.tools import flatbuffer_utils

    input_path = tmp_path / "input.tflite"
    output_path = tmp_path / "output.tflite"
    _write_int8_bias_conv_model(input_path)

    rewritten = rewrite_int8_conv_biases_to_int32(input_path, output_path)

    model = flatbuffer_utils.read_model(output_path)
    bias = model.subgraphs[0].tensors[2]
    values = np.frombuffer(bytes(model.buffers[bias.buffer].data), dtype=np.int32)

    assert rewritten == 1
    assert bias.type == schema.TensorType.INT32
    np.testing.assert_allclose(bias.quantization.scale, np.array([0.125, 0.0625]))
    assert bias.quantization.zeroPoint.tolist() == [0, 0]
    assert values.tolist() == [10, -80]
