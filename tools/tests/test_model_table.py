from pathlib import Path

from tools.flash_runner.model_table import build_model_table
from tools.model_converter.model_registry import ModelSpec


def test_build_model_table_layout() -> None:
    model = ModelSpec(
        name="yolo11n_od_192",
        task="object_detection",
        source={"type": "ultralytics", "model": "yolo11n.pt"},
        input_shape=[1, 3, 192, 192],
        calibration_dataset="coco_128",
        tier=1,
        flash_address=0x400000,
        model_zoo_ref="tflm_yolo11_od",
    )

    payload = build_model_table([(model, 4532736)])

    assert payload[:4] == b"MODL"
    assert int.from_bytes(payload[8:12], "little") == 1
    assert payload[16:48].rstrip(b"\x00") == b"yolo11n_od_192"
    assert payload[48:80].rstrip(b"\x00") == b"object_detection"


def test_build_model_table_accepts_3d_audio_feature_shape() -> None:
    model = ModelSpec(
        name="honk_res8_narrow_kws_101x40",
        task="keyword_spotting",
        source={"type": "honk", "architecture": "res8-narrow"},
        input_shape=[1, 101, 40],
        calibration_dataset="speech_commands",
        tier=1,
        flash_address=0xA00000,
    )

    payload = build_model_table([(model, 36000)])

    assert int.from_bytes(payload[80:84], "little") == 0xA00000
    assert int.from_bytes(payload[88:92], "little") == 1
    assert int.from_bytes(payload[92:96], "little") == 101
    assert int.from_bytes(payload[96:100], "little") == 40


def test_firmware_uses_expected_model_table_address() -> None:
    header = Path(
        "EPII_CM55M_APP_S/app/scenario_app/model_benchmark/common_config.h"
    ).read_text(encoding="utf-8")
    assert "#define MODEL_TABLE_FLASH_ADDR    0x3A200000" in header
    assert "#define MAX_MODEL_COUNT           20" in header
