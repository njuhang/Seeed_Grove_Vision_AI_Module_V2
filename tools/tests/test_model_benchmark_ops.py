"""Firmware contract test: the benchmark app must register a comprehensive
TFLM op set so that models with CPU fallback ops (not 100% Vela-compiled) can
also run, per spec section 4.8. The firmware C++ has no host test harness, so we
assert against the source the same way test_model_table.py does."""
import re
from pathlib import Path

FIRMWARE_CPP = Path(
    "EPII_CM55M_APP_S/app/scenario_app/model_benchmark/cvapp_model_benchmark.cpp"
)

# Spec 4.8 op set. ReduceMin is intentionally excluded: this TFLM build only
# ships AddMinimum (element-wise), not AddReduceMin. BatchMatMul is kept because
# the working yolo11n model (yolo11n_full_integer_quant_192_..._batch_matmul_vela)
# needs it after Vela compilation.
REQUIRED_OPS = [
    "AddAdd",
    "AddBroadcastTo",
    "AddConv2D",
    "AddDepthwiseConv2D",
    "AddDequantize",
    "AddReshape",
    "AddSoftmax",
    "AddTranspose",
    "AddMaxPool2D",
    "AddAveragePool2D",
    "AddFullyConnected",
    "AddConcatenation",
    "AddMul",
    "AddRelu",
    "AddResizeBilinear",
    "AddResizeNearestNeighbor",
    "AddSplit",
    "AddPad",
    "AddPrelu",
    "AddStridedSlice",
    "AddGather",
    "AddGatherNd",
    "AddExp",
    "AddLog",
    "AddLeakyRelu",
    "AddHardSwish",
    "AddQuantize",
    "AddReduceMax",
    "AddSum",
    "AddMean",
    "AddArgMax",
    "AddEthosU",
    "AddBatchMatMul",
    "AddL2Normalization",
]


def test_firmware_registers_comprehensive_op_set() -> None:
    source = FIRMWARE_CPP.read_text(encoding="utf-8")

    missing = [op for op in REQUIRED_OPS if f"{op}(" not in source]

    assert not missing, f"op resolver missing registrations: {missing}"


def test_op_resolver_capacity_fits_required_ops() -> None:
    source = FIRMWARE_CPP.read_text(encoding="utf-8")

    match = re.search(r"MicroMutableOpResolver<(\d+)>", source)
    assert match is not None, "MicroMutableOpResolver template not found"

    capacity = int(match.group(1))
    assert capacity >= len(REQUIRED_OPS), (
        f"resolver capacity {capacity} < required op count {len(REQUIRED_OPS)}"
    )


def test_random_input_fill_handles_float_and_int8_tensors() -> None:
    source = FIRMWARE_CPP.read_text(encoding="utf-8")

    assert "input_tensor->type" in source
    assert "kTfLiteFloat32" in source
    assert "kTfLiteInt8" in source



def test_fully_connected_common_allows_dynamic_filter_zero_points() -> None:
    source = Path(
        "EPII_CM55M_APP_S/library/inference/tflmtag2412_u55tag2411/"
        "tensorflow/lite/micro/kernels/fully_connected_common.cc"
    ).read_text(encoding="utf-8")

    assert "TFLITE_DCHECK(filter->params.zero_point == 0)" not in source


def test_cmsis_fully_connected_conv1x1_fast_path_requires_symmetric_filter() -> None:
    source = Path(
        "EPII_CM55M_APP_S/library/inference/tflmtag2412_u55tag2411/"
        "tensorflow/lite/micro/kernels/cmsis_nn/fully_connected.cc"
    ).read_text(encoding="utf-8")

    assert "data->reference_op_data.filter_zero_point == 0" in source
