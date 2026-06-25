import json

from tools.report.generate_report import build_rows
from tools.report.generate_report import load_conversion_summary
from tools.report.generate_report import load_payload
from tools.report.generate_report import load_vela_infos
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


def test_load_vela_infos_accepts_utf8_bom(tmp_path) -> None:
    vela_info = tmp_path / "yolo11s_od_192.vela_info.json"
    vela_info.write_text(
        "\ufeff" + json.dumps({"npu_utilization_pct": 98.5, "cpu_fallback_ops": ["Passthrough"]}),
        encoding="utf-8",
    )

    infos = load_vela_infos(tmp_path)

    assert infos["yolo11s_od_192"]["npu_utilization_pct"] == 98.5
    assert infos["yolo11s_od_192"]["cpu_fallback_ops"] == ["Passthrough"]


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



def test_merge_result_row_preserves_board_status_reason() -> None:
    board_result = {
        "name": "segformerb0_seg_64",
        "task": "semantic_segmentation",
        "model_size_bytes": 999248,
        "arena_used_bytes": 0,
        "latency_ms": {"avg": 0.0, "min": 0.0, "max": 0.0},
        "status": "load_failed",
        "status_reason": "AllocateTensors requested 4,304,384 bytes; available 1,811,520.",
    }

    row = merge_result_row(board_result, {})

    assert row["Status"] == "load_failed"
    assert "AllocateTensors requested" in row["Reason"]


def test_merge_result_row_handles_flash_failure_without_latency() -> None:
    board_result = {
        "name": "deeplabv3_mbnv3_seg_320",
        "task": "segmentation",
        "model_size_bytes": 36339088,
        "status": "flash_failed",
        "status_reason": "xmodem_send.py exited with status 1",
    }

    row = merge_result_row(board_result, {"npu_utilization_pct": 99.5, "cpu_fallback_ops": []})

    assert row["Status"] == "flash_failed"
    assert row["Arena (KB)"] == "-"
    assert row["Avg (ms)"] == "-"
    assert row["Reason"] == "xmodem_send.py exited with status 1"



def test_load_payload_accepts_utf8_bom(tmp_path) -> None:
    payload_path = tmp_path / "benchmark_result.json"
    payload_path.write_text(
        "\ufeff" + json.dumps({"benchmark": {"device": "himax_hx6538"}, "models": []}),
        encoding="utf-8",
    )

    payload = load_payload(payload_path)

    assert payload["benchmark"]["device"] == "himax_hx6538"


def test_load_conversion_summary_accepts_utf8_bom(tmp_path) -> None:
    summary_path = tmp_path / "conversion_summary.json"
    summary_path.write_text(
        "\ufeff" + json.dumps({"results": [{"name": "deeplabv3_mbnv3_seg_320"}]}),
        encoding="utf-8",
    )

    results = load_conversion_summary(summary_path)

    assert results[0]["name"] == "deeplabv3_mbnv3_seg_320"
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


def test_build_rows_treats_exported_from_source_as_success_without_failure_row() -> None:
    payload = {"models": []}
    conversion_results = [
        {
            "name": "yolo11n_od_192",
            "task": "object_detection",
            "status": "exported_from_source",
            "notes": [],
            "missing_dependencies": [],
        }
    ]

    rows = build_rows(
        payload,
        manifest_tasks={"yolo11n_od_192": "object_detection"},
        vela_infos={},
        conversion_results=conversion_results,
    )

    assert rows == []


def test_build_rows_surfaces_npu_optimization_summary() -> None:
    payload = {"models": [_ok_board_result("retinaface_mnet_face_160")]}
    rows = build_rows(
        payload,
        manifest_tasks={"retinaface_mnet_face_160": "face_detection"},
        vela_infos={
            "retinaface_mnet_face_160": {
                "npu_utilization_pct": 100.0,
                "cpu_fallback_ops": [],
            }
        },
        conversion_results=[
            {
                "name": "retinaface_mnet_face_160",
                "task": "face_detection",
                "status": "exported_from_source",
                "notes": [],
                "missing_dependencies": [],
                "npu_optimization": {
                    "applied": True,
                    "best_candidate": "all_safe_rewrites",
                    "baseline_summary": {
                        "npu_utilization_pct": 39.4,
                        "cpu_fallback_ops": ["Passthrough"],
                    },
                    "optimized_summary": {
                        "npu_utilization_pct": 100.0,
                        "cpu_fallback_ops": [],
                    },
                },
            }
        ],
    )

    row = rows[0]
    assert row["NPU Optimization"] == "all_safe_rewrites: 39.4->100.0%; fallback Passthrough->-"


def test_build_rows_marks_optimized_board_failure_with_baseline_fallback() -> None:
    board_result = _ok_board_result("retinaface_mnet_face_160")
    board_result["status"] = "load_failed"
    board_result["status_reason"] = "optimized candidate failed AllocateTensors"

    rows = build_rows(
        {"models": [board_result]},
        manifest_tasks={"retinaface_mnet_face_160": "face_detection"},
        vela_infos={},
        conversion_results=[
            {
                "name": "retinaface_mnet_face_160",
                "task": "face_detection",
                "status": "exported_from_source",
                "notes": [],
                "missing_dependencies": [],
                "npu_optimization": {
                    "applied": True,
                    "best_candidate": "all_safe_rewrites",
                    "baseline_vela_path": "artifacts/retinaface_mnet_face_160/baseline_vela.tflite",
                    "optimized_vela_path": "artifacts/retinaface_mnet_face_160/optimized_vela.tflite",
                },
            }
        ],
    )

    assert rows[0]["Status"] == "npu_opt_board_failed"
    assert "baseline_vela.tflite" in rows[0]["Reason"]


def test_build_rows_appends_model_zoo_comparison_diff_row() -> None:
    payload = {
        "models": [
            {
                "name": "yolo11n_od_192 (ours)",
                "task": "object_detection",
                "model_size_bytes": 2048000,
                "arena_used_bytes": 1048576,
                "latency_ms": {"avg": 84.8, "min": 82.0, "max": 91.5},
                "status": "ok",
            },
            {
                "name": "yolo11n_od_192 (model_zoo)",
                "task": "object_detection",
                "model_size_bytes": 2050048,
                "arena_used_bytes": 1048576,
                "latency_ms": {"avg": 85.1, "min": 82.3, "max": 91.9},
                "status": "ok",
            },
        ]
    }

    rows = build_rows(
        payload,
        manifest_tasks={"yolo11n_od_192": "object_detection"},
        vela_infos={
            "yolo11n_od_192": {"npu_utilization_pct": 100.0, "cpu_fallback_ops": []},
            "yolo11n_ref": {"npu_utilization_pct": 98.0, "cpu_fallback_ops": ["Passthrough"]},
        },
        conversion_results=[
            {
                "name": "yolo11n_od_192",
                "status": "exported_from_source",
                "model_zoo_ref": "yolo11n_ref.tflite",
            }
        ],
    )

    names = [row["Model"] for row in rows]
    assert "yolo11n_od_192 (ours)" in names
    assert "yolo11n_od_192 (model_zoo)" in names
    assert "yolo11n_od_192 (diff)" in names
    diff = next(row for row in rows if row["Model"] == "yolo11n_od_192 (diff)")
    assert diff["Task"] == "object_detection"
    assert diff["Size (KB)"] == -2.0
    assert diff["Arena (KB)"] == 0.0
    assert diff["Avg (ms)"] == -0.3
    assert diff["NPU%"] == 2.0
    assert diff["CPU Fallback"] == "ours:- | model_zoo:Passthrough"
    assert diff["Status"] == "comparison"


def test_build_rows_appends_yolo_size_trend_row() -> None:
    payload = {
        "models": [
            {
                "name": "yolo11n_od_192",
                "task": "object_detection",
                "model_size_bytes": 1024 * 1000,
                "arena_used_bytes": 1024 * 900,
                "latency_ms": {"avg": 80.0, "min": 79.0, "max": 81.0},
                "status": "ok",
            },
            {
                "name": "yolo11s_od_192",
                "task": "object_detection",
                "model_size_bytes": 1024 * 2000,
                "arena_used_bytes": 1024 * 1100,
                "latency_ms": {"avg": 130.0, "min": 128.0, "max": 134.0},
                "status": "ok",
            },
        ]
    }

    rows = build_rows(
        payload,
        manifest_tasks={
            "yolo11n_od_192": "object_detection",
            "yolo11s_od_192": "object_detection",
        },
        vela_infos={
            "yolo11n_od_192": {"npu_utilization_pct": 98.0, "cpu_fallback_ops": []},
            "yolo11s_od_192": {"npu_utilization_pct": 96.0, "cpu_fallback_ops": ["Passthrough"]},
        },
    )

    trend = next(row for row in rows if row["Model"] == "yolo11_od_192 (size trend)")
    assert trend["Status"] == "trend"
    assert trend["Size (KB)"] == "n:1000.0, s:2000.0"
    assert trend["Avg (ms)"] == "n:80.0, s:130.0"
    assert trend["NPU%"] == "n:98.0, s:96.0"


def test_render_markdown_includes_model_zoo_diff_row() -> None:
    markdown = render_markdown(
        [
            {
                "Model": "yolo11n_od_192 (diff)",
                "Task": "object_detection",
                "Size (KB)": -2.0,
                "Arena (KB)": 0.0,
                "Avg (ms)": -0.3,
                "Min (ms)": -0.3,
                "Max (ms)": -0.4,
                "NPU%": 2.0,
                "CPU Fallback": "ours:- | model_zoo:Passthrough",
                "Status": "comparison",
                "Reason": "ours - model_zoo",
            }
        ],
        "Himax HX6538 Model Benchmark Report",
    )

    assert "yolo11n_od_192 (diff)" in markdown
    assert "ours:- \\| model_zoo:Passthrough" in markdown
    assert " | comparison | ours - model_zoo |" in markdown


def test_render_markdown_escapes_pipe_characters_inside_cells() -> None:
    markdown = render_markdown(
        [
            {
                "Model": "mobilenetv2_cls_224 (diff)",
                "Task": "classification",
                "Size (KB)": 1652.9,
                "Arena (KB)": 1098.1,
                "Avg (ms)": 13.9,
                "Min (ms)": 13.9,
                "Max (ms)": 13.9,
                "NPU%": "-",
                "CPU Fallback": "ours:Passthrough | model_zoo:-",
                "Status": "comparison",
                "Reason": "ours - model_zoo",
            }
        ],
        "Himax HX6538 Model Benchmark Report",
    )

    assert "ours:Passthrough \\| model_zoo:-" in markdown


def test_build_rows_treats_degenerate_recompiled_model_zoo_vela_info_as_unavailable() -> None:
    payload = {
        "models": [
            {
                "name": "mobilenetv2_cls_224 (model_zoo)",
                "task": "classification",
                "model_size_bytes": 1704672,
                "arena_used_bytes": 385748,
                "latency_ms": {"avg": 103.937, "min": 103.937, "max": 103.938},
                "status": "ok",
            }
        ]
    }

    rows = build_rows(
        payload,
        manifest_tasks={"mobilenetv2_cls_224": "classification"},
        vela_infos={
            "qat_pruning_model_vela": {
                "cpu_ops": 1,
                "npu_ops": 0,
                "npu_utilization_pct": 0.0,
                "cpu_fallback_ops": ["Passthrough"],
                "input_model": "/repo/model_zoo/tflm_mb_cls/qat_pruning_model_vela.tflite",
            }
        },
        conversion_results=[
            {
                "name": "mobilenetv2_cls_224",
                "status": "exported_from_source",
                "model_zoo_ref": "qat_pruning_model_vela.tflite",
            }
        ],
    )

    row = rows[0]
    assert row["Model"] == "mobilenetv2_cls_224 (model_zoo)"
    assert row["NPU%"] == "-"
    assert row["CPU Fallback"] == "-"


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
