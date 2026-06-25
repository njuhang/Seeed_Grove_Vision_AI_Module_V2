import json
from pathlib import Path

from tools.report.generate_phase6_summary import build_rows
from tools.report.generate_phase6_summary import render_markdown
from tools.report.generate_phase6_summary import write_summary


def test_build_rows_merges_plan_conversion_and_vela_info(tmp_path: Path) -> None:
    int8_path = tmp_path / "model_int8.tflite"
    vela_path = tmp_path / "model_vela.tflite"
    int8_path.write_bytes(b"a" * 2048)
    vela_path.write_bytes(b"b" * 1024)
    vela_dir = tmp_path / "vela"
    vela_dir.mkdir()
    (vela_dir / "yolo_fastestv2_od_192.vela_info.json").write_text(
        json.dumps({"npu_utilization_pct": 100.0, "cpu_fallback_ops": []}),
        encoding="utf-8",
    )

    rows = build_rows(
        {
            "batches": [
                {
                    "batch_index": 0,
                    "models": [
                        {
                            "name": "yolo_fastestv2_od_192",
                            "flash_address": "0xc0d000",
                        }
                    ],
                }
            ]
        },
        {
            "results": [
                {
                    "name": "yolo_fastestv2_od_192",
                    "task": "object_detection",
                    "source_type": "yolo_fastestv2",
                    "status": "exported_from_source",
                    "int8_model_path": str(int8_path),
                    "vela_model_path": str(vela_path),
                }
            ]
        },
        vela_dir,
    )

    assert rows == [
        {
            "Model": "yolo_fastestv2_od_192",
            "Task": "object_detection",
            "Source": "yolo_fastestv2",
            "Status": "exported_from_source",
            "Batch": 0,
            "Flash": "0xc0d000",
            "Int8 KB": 2.0,
            "Vela KB": 1.0,
            "NPU%": 100.0,
            "CPU Fallback": "-",
        }
    ]


def test_render_markdown_includes_phase6_columns() -> None:
    markdown = render_markdown(
        [
            {
                "Model": "yolo11s_od_192",
                "Task": "object_detection",
                "Source": "ultralytics",
                "Status": "exported_from_source",
                "Batch": 0,
                "Flash": "0x400000",
                "Int8 KB": 9796.7,
                "Vela KB": 8243.3,
                "NPU%": 98.5,
                "CPU Fallback": "Passthrough",
            }
        ],
        "Phase 6",
    )

    assert "| Model | Task | Source | Status | Batch | Flash |" in markdown
    assert "yolo11s_od_192" in markdown
    assert "Passthrough" in markdown


def test_write_summary_writes_markdown(tmp_path: Path) -> None:
    int8_path = tmp_path / "model_int8.tflite"
    vela_path = tmp_path / "model_vela.tflite"
    int8_path.write_bytes(b"a")
    vela_path.write_bytes(b"b")
    plan_path = tmp_path / "benchmark_plan.json"
    conversion_path = tmp_path / "conversion_summary.json"
    vela_dir = tmp_path / "vela"
    vela_dir.mkdir()
    output_path = tmp_path / "phase6_summary.md"

    plan_path.write_text(
        json.dumps(
            {
                "batches": [
                    {
                        "batch_index": 0,
                        "models": [{"name": "demo", "flash_address": "0x400000"}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    conversion_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "name": "demo",
                        "task": "object_detection",
                        "source_type": "demo",
                        "status": "exported_from_source",
                        "int8_model_path": str(int8_path),
                        "vela_model_path": str(vela_path),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (vela_dir / "demo.vela_info.json").write_text(
        json.dumps({"npu_utilization_pct": 99.0, "cpu_fallback_ops": ["RESHAPE"]}),
        encoding="utf-8",
    )

    result = write_summary(plan_path, conversion_path, vela_dir, output_path)

    assert result == output_path
    assert "demo" in output_path.read_text(encoding="utf-8")
