import json

from tools.report.generate_report import build_rows
from tools.report.generate_report import merge_failure_row
from tools.report.generate_report import merge_result_row
from tools.report.generate_report import render_markdown
from tools.report.generate_report import write_report


def _ok_board_result(name: str = "yolo11n_od_192") -> dict:
    return {
        "name": name,
        "task": "object_detection",
        "model_size_bytes": 2041472,
        "arena_used_bytes": 1086052,
        "latency_ms": {"avg": 83, "min": 83, "max": 83},
        "status": "ok",
    }


def _manifest_text() -> str:
    return (
        "models:\n"
        "  - name: yolo11n_od_192\n"
        "    task: object_detection\n"
        "    source: {type: ultralytics, model: yolo11n.pt}\n"
        "    input_shape: [1, 3, 192, 192]\n"
        "    calibration_dataset: coco_128\n"
        "    tier: 1\n"
        "    flash_address: 0x00400000\n"
        "  - name: squeezenet11_cls_224\n"
        "    task: classification\n"
        "    source: {type: torchvision, model: squeezenet1_1}\n"
        "    input_shape: [1, 3, 224, 224]\n"
        "    calibration_dataset: imagenet_1k_random\n"
        "    tier: 1\n"
        "    flash_address: 0x00800000\n"
    )


def test_merge_result_row() -> None:
    board_result = {
        "name": "yolo11n_od_192",
        "task": "object_detection",
        "model_size_bytes": 4532736,
        "arena_used_bytes": 1048576,
        "latency_ms": {"avg": 85.3, "min": 82.1, "max": 91.7},
        "status": "ok",
    }
    vela_info = {
        "npu_utilization_pct": 95.0,
        "cpu_fallback_ops": ["RESHAPE", "TRANSPOSE"],
    }

    row = merge_result_row(board_result, vela_info)

    assert row["Model"] == "yolo11n_od_192"
    assert row["NPU%"] == 95.0
    assert row["CPU Fallback"] == "RESHAPE,TRANSPOSE"


def test_merge_result_row_renders_high_npu_and_single_fallback_op() -> None:
    # Regression for the vela 5.1.0 data path: parse_vela_output feeds
    # npu_utilization_pct / cpu_fallback_ops into merge_result_row; this locks
    # the exact rendering for the high-NPU + non-empty-fallback case.
    board_result = {
        "name": "yolo11n_od_192",
        "task": "object_detection",
        "model_size_bytes": 4532736,
        "arena_used_bytes": 1048576,
        "latency_ms": {"avg": 85.3, "min": 82.1, "max": 91.7},
        "status": "ok",
    }
    vela_info = {
        "npu_utilization_pct": 95.0,
        "cpu_fallback_ops": ["Passthrough"],
    }

    row = merge_result_row(board_result, vela_info)

    assert row["NPU%"] == 95.0
    assert row["CPU Fallback"] == "Passthrough"


def test_merge_result_row_renders_empty_fallback_as_dash() -> None:
    # A fully-offloaded model has cpu_fallback_ops == [] and must render the
    # CPU Fallback cell as "-", not an empty string.
    board_result = {
        "name": "yolo11n_od_192",
        "task": "object_detection",
        "model_size_bytes": 4532736,
        "arena_used_bytes": 1048576,
        "latency_ms": {"avg": 60.0, "min": 59.0, "max": 61.0},
        "status": "ok",
    }
    vela_info = {"npu_utilization_pct": 100.0, "cpu_fallback_ops": []}

    row = merge_result_row(board_result, vela_info)

    assert row["NPU%"] == 100.0
    assert row["CPU Fallback"] == "-"


def test_merge_result_row_renders_missing_vela_info_as_dashes() -> None:
    # When no vela_info is available (empty dict from vela_infos.get(name, {})),
    # both columns must fall back to "-" via .get(..., "-") / `or "-"`.
    board_result = {
        "name": "yolo11n_od_192",
        "task": "object_detection",
        "model_size_bytes": 4532736,
        "arena_used_bytes": 1048576,
        "latency_ms": {"avg": 70.0, "min": 69.0, "max": 71.0},
        "status": "ok",
    }

    row = merge_result_row(board_result, {})

    assert row["NPU%"] == "-"
    assert row["CPU Fallback"] == "-"


def test_render_markdown_emits_npu_and_cpu_fallback_cells() -> None:
    # End-to-end check one layer up: render_markdown must surface both columns
    # in the table body for a high-NPU + non-empty-fallback row.
    board_result = {
        "name": "yolo11n_od_192",
        "task": "object_detection",
        "model_size_bytes": 4532736,
        "arena_used_bytes": 1048576,
        "latency_ms": {"avg": 85.3, "min": 82.1, "max": 91.7},
        "status": "ok",
    }
    vela_info = {"npu_utilization_pct": 63.0, "cpu_fallback_ops": ["Passthrough"]}
    row = merge_result_row(board_result, vela_info)

    markdown = render_markdown([row], "Himax HX6538 Model Benchmark Report")

    assert "| NPU% | CPU Fallback |" in markdown
    assert " | 63.0 | Passthrough | " in markdown


def test_merge_result_row_uses_manifest_task_when_board_task_unknown() -> None:
    row = merge_result_row(
        {
            "name": "yolov8n_od_192",
            "task": "unknown",
            "model_size_bytes": 3984588,
            "arena_used_bytes": 983040,
            "latency_ms": {"avg": 79.5, "min": 77.2, "max": 83.9},
            "status": "ok",
        },
        {"npu_utilization_pct": 100.0, "cpu_fallback_ops": []},
        manifest_task="object_detection",
    )

    assert row["Task"] == "object_detection"
    assert row["CPU Fallback"] == "-"


def test_merge_failure_row_marks_status_and_reason() -> None:
    conversion = {
        "name": "squeezenet11_cls_224",
        "task": "classification",
        "status": "backend_not_implemented",
        "notes": ["the real export backend is not implemented yet"],
        "missing_dependencies": [],
    }

    row = merge_failure_row(conversion)

    assert row["Model"] == "squeezenet11_cls_224"
    assert row["Task"] == "classification"
    assert row["Status"] == "backend_not_implemented"
    assert "not implemented" in row["Reason"]
    # A model that never ran has no latency / arena numbers.
    assert row["Avg (ms)"] == "-"
    assert row["Size (KB)"] == "-"


def test_merge_failure_row_lists_missing_dependencies() -> None:
    conversion = {
        "name": "mobilenetv2_cls_224",
        "task": "classification",
        "status": "blocked_missing_wsl_dependencies",
        "notes": ["wsl conversion environment is missing required packages"],
        "missing_dependencies": ["torch", "ultralytics"],
    }

    row = merge_failure_row(conversion)

    assert "torch" in row["Reason"]
    assert "ultralytics" in row["Reason"]


def test_build_rows_appends_failure_rows_for_unsuccessful_models() -> None:
    payload = {"models": [_ok_board_result("yolo11n_od_192")]}
    conversion_results = [
        {
            "name": "yolo11n_od_192",
            "status": "staged_model_zoo_ref",
            "notes": [],
            "missing_dependencies": [],
        },
        {
            "name": "squeezenet11_cls_224",
            "task": "classification",
            "status": "backend_not_implemented",
            "notes": ["not implemented yet"],
            "missing_dependencies": [],
        },
    ]

    rows = build_rows(
        payload,
        manifest_tasks={
            "yolo11n_od_192": "object_detection",
            "squeezenet11_cls_224": "classification",
        },
        vela_infos={},
        conversion_results=conversion_results,
    )

    names = [row["Model"] for row in rows]
    assert "yolo11n_od_192" in names
    assert "squeezenet11_cls_224" in names
    failure = next(row for row in rows if row["Model"] == "squeezenet11_cls_224")
    assert failure["Status"] == "backend_not_implemented"


def test_build_rows_does_not_duplicate_model_that_ran_on_board() -> None:
    payload = {"models": [_ok_board_result("yolo11n_od_192")]}
    conversion_results = [
        {
            "name": "yolo11n_od_192",
            "task": "object_detection",
            "status": "backend_not_implemented",
            "notes": ["x"],
            "missing_dependencies": [],
        }
    ]

    rows = build_rows(
        payload,
        manifest_tasks={"yolo11n_od_192": "object_detection"},
        vela_infos={},
        conversion_results=conversion_results,
    )

    assert sum(1 for r in rows if r["Model"] == "yolo11n_od_192") == 1


def test_write_report_marks_conversion_failures_with_reason(tmp_path) -> None:
    manifest = tmp_path / "models.yaml"
    manifest.write_text(_manifest_text(), encoding="utf-8")
    summary_path = tmp_path / "conversion_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "name": "squeezenet11_cls_224",
                        "task": "classification",
                        "status": "backend_not_implemented",
                        "notes": ["the real export backend is not implemented yet"],
                        "missing_dependencies": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    payload = {"benchmark": {"device": "himax_hx6538"}, "models": [_ok_board_result("yolo11n_od_192")]}

    md_path, _ = write_report(
        payload,
        manifest,
        tmp_path / "benchmark_report",
        conversion_summary_path=summary_path,
    )

    markdown = md_path.read_text(encoding="utf-8")
    assert "squeezenet11_cls_224" in markdown
    assert "backend_not_implemented" in markdown
    assert "Reason" in markdown
