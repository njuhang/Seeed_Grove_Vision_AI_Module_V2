from pathlib import Path


DEQUANTIZE_COMMON = Path(
    "EPII_CM55M_APP_S/library/inference/tflmtag2412_u55tag2411/"
    "tensorflow/lite/micro/kernels/dequantize_common.cc"
)
DEQUANTIZE_EVAL = Path(
    "EPII_CM55M_APP_S/library/inference/tflmtag2412_u55tag2411/"
    "tensorflow/lite/micro/kernels/dequantize.cc"
)


def test_dequantize_prepare_accepts_float16_to_float32() -> None:
    source = DEQUANTIZE_COMMON.read_text(encoding="utf-8")

    assert "input->type == kTfLiteFloat16" in source
    assert "output->type == kTfLiteFloat32" in source


def test_dequantize_eval_converts_float16_inputs() -> None:
    source = DEQUANTIZE_EVAL.read_text(encoding="utf-8")

    assert "case kTfLiteFloat16:" in source
    assert "Float16ToFloat32" in source
    assert "GetTensorData<TfLiteFloat16>(input)" in source
