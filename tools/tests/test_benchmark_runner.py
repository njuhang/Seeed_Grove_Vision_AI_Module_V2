import json
from pathlib import Path

from unittest.mock import patch

from tools.benchmark_runner import capture_board_result, compute_batches, main, parse_args, parse_tiers, prepare_artifacts, resolve_model_artifacts, run_board_benchmark, run_execution_plan, write_execution_plan


def test_compute_batches() -> None:
    batches = compute_batches(
        [
            ("yolo11n_od_192", 0x00400000, 0x00160000),
            ("yolov8n_od_192", 0x00600000, 0x00150000),
            ("mobilenetv2_cls_224", 0x00800000, 0x000F0000),
        ],
        flash_capacity=0x00400000,
    )

    assert len(batches) == 1
    assert [entry[0] for entry in batches[0]] == [
        "yolo11n_od_192",
        "yolov8n_od_192",
        "mobilenetv2_cls_224",
    ]


def test_compute_batches_uses_real_model_sizes_not_nominal_slot_starts() -> None:
    batches = compute_batches(
        [
            ("mobilenetv3small_cls_224", 0x00A00000, 2_506_400),
            ("yolov5n_od_192", 0x00C00000, 1_768_144),
        ],
        flash_capacity=0x00400000,
    )

    assert len(batches) == 2
    assert [entry[0] for entry in batches[0]] == ["mobilenetv3small_cls_224"]
    assert [entry[0] for entry in batches[1]] == ["yolov5n_od_192"]


def test_resolve_model_artifacts_prefers_generated_artifacts(tmp_path: Path) -> None:
    manifest = tmp_path / "models.yaml"
    manifest.write_text(
        """
models:
  - name: yolo11n_od_192
    task: object_detection
    source:
      type: ultralytics
      model: yolo11n.pt
    input_shape: [1, 3, 192, 192]
    calibration_dataset: coco_128
    tier: 1
    flash_address: 0x00400000
    model_zoo_ref: yolo11n_vela.tflite
        """.strip(),
        encoding="utf-8",
    )
    artifact_path = tmp_path / "artifacts" / "yolo11n_od_192" / "yolo11n_od_192_vela.tflite"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(b"generated")
    model_path = tmp_path / "model_zoo" / "tflm_yolo11_od" / "yolo11n_vela.tflite"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"abcd")

    resolved = resolve_model_artifacts(tmp_path, manifest)

    assert len(resolved) == 1
    assert resolved[0].model_path == artifact_path
    assert resolved[0].model_size_bytes == len(b"generated")


def test_resolve_model_artifacts_prefers_optimized_artifact_when_baseline_is_preserved(tmp_path: Path) -> None:
    manifest = tmp_path / "models.yaml"
    manifest.write_text(
        """
models:
  - name: retinaface_mnet_face_160
    task: face_detection
    source:
      type: retinaface
      repo_path: artifacts_debug/vendor/retinaface
      weights: weights/mobilenet0.25_Final.pth
    input_shape: [1, 3, 160, 160]
    calibration_dataset: coco_128
    tier: 1
    flash_address: 0x02400000
        """.strip(),
        encoding="utf-8",
    )
    artifact_dir = tmp_path / "artifacts" / "retinaface_mnet_face_160"
    artifact_dir.mkdir(parents=True)
    baseline = artifact_dir / "retinaface_mnet_face_160_vela.tflite"
    optimized = artifact_dir / "retinaface_mnet_face_160_vela_optimized.tflite"
    baseline.write_bytes(b"baseline")
    optimized.write_bytes(b"optimized")

    resolved = resolve_model_artifacts(tmp_path, manifest)

    assert len(resolved) == 1
    assert resolved[0].model_path == optimized
    assert resolved[0].model_size_bytes == len(b"optimized")


def test_resolve_model_artifacts_falls_back_to_model_zoo_refs(tmp_path: Path) -> None:
    manifest = tmp_path / "models.yaml"
    manifest.write_text(
        """
models:
  - name: yolo11n_od_192
    task: object_detection
    source:
      type: ultralytics
      model: yolo11n.pt
    input_shape: [1, 3, 192, 192]
    calibration_dataset: coco_128
    tier: 1
    flash_address: 0x00400000
    model_zoo_ref: yolo11n_vela.tflite
        """.strip(),
        encoding="utf-8",
    )
    model_path = tmp_path / "model_zoo" / "tflm_yolo11_od" / "yolo11n_vela.tflite"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"abcd")

    resolved = resolve_model_artifacts(tmp_path, manifest)

    assert len(resolved) == 1
    assert resolved[0].model_path == model_path
    assert resolved[0].model_size_bytes == 4


def test_resolve_model_artifacts_can_include_tier2(tmp_path: Path) -> None:
    manifest = tmp_path / "models.yaml"
    manifest.write_text(
        """
models:
  - name: yolo11n_od_192
    task: object_detection
    source:
      type: ultralytics
      model: yolo11n.pt
    input_shape: [1, 3, 192, 192]
    calibration_dataset: coco_128
    tier: 1
    flash_address: 0x00400000
  - name: yolo11s_od_192
    task: object_detection
    source:
      type: ultralytics
      model: yolo11s.pt
    input_shape: [1, 3, 192, 192]
    calibration_dataset: coco_128
    tier: 2
    flash_address: 0x00600000
        """.strip(),
        encoding="utf-8",
    )
    artifact_path = tmp_path / "artifacts" / "yolo11s_od_192" / "yolo11s_od_192_vela.tflite"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(b"tier2")

    resolved = resolve_model_artifacts(tmp_path, manifest, tiers={2})

    assert [item.spec.name for item in resolved] == ["yolo11s_od_192"]


def test_write_execution_plan_skips_models_without_resolved_artifacts(tmp_path: Path) -> None:
    manifest = tmp_path / "models.yaml"
    manifest.write_text(
        """
models:
  - name: generated_model
    task: classification
    source:
      type: torchvision
      model: mobilenet_v2
    input_shape: [1, 3, 224, 224]
    calibration_dataset: imagenet_1k_random
    tier: 1
    flash_address: 0x00800000
  - name: failed_model
    task: classification
    source:
      type: torchvision
      model: mobilenet_v3_small
    input_shape: [1, 3, 224, 224]
    calibration_dataset: imagenet_1k_random
    tier: 1
    flash_address: 0x00A00000
        """.strip(),
        encoding="utf-8",
    )
    output_dir = tmp_path / "plan"
    artifact_path = output_dir / "generated_model" / "generated_model_vela.tflite"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(b"generated")

    plan_path = write_execution_plan(
        repo_root=tmp_path,
        manifest_path=manifest,
        output_dir=output_dir,
        flash_capacity=0x00400000,
    )

    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    assert payload["batch_count"] == 1
    assert [model["name"] for model in payload["batches"][0]["models"]] == ["generated_model"]


def test_write_execution_plan_records_selected_tiers(tmp_path: Path) -> None:
    manifest = tmp_path / "models.yaml"
    manifest.write_text(
        """
models:
  - name: yolo11s_od_192
    task: object_detection
    source:
      type: ultralytics
      model: yolo11s.pt
    input_shape: [1, 3, 192, 192]
    calibration_dataset: coco_128
    tier: 2
    flash_address: 0x00600000
        """.strip(),
        encoding="utf-8",
    )
    output_dir = tmp_path / "plan"
    artifact_path = output_dir / "yolo11s_od_192" / "yolo11s_od_192_vela.tflite"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(b"tier2")

    plan_path = write_execution_plan(
        repo_root=tmp_path,
        manifest_path=manifest,
        output_dir=output_dir,
        flash_capacity=0x00400000,
        tiers={2},
    )

    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    assert payload["tiers"] == [2]
    assert payload["batches"][0]["models"][0]["name"] == "yolo11s_od_192"


def test_write_execution_plan_uses_current_output_dir_artifacts(tmp_path: Path) -> None:
    manifest = tmp_path / "models.yaml"
    manifest.write_text(
        """
models:
  - name: generated_model
    task: classification
    source:
      type: torchvision
      model: mobilenet_v2
    input_shape: [1, 3, 224, 224]
    calibration_dataset: imagenet_1k_random
    tier: 1
    flash_address: 0x00800000
        """.strip(),
        encoding="utf-8",
    )
    stale_artifact = tmp_path / "artifacts" / "generated_model" / "generated_model_vela.tflite"
    stale_artifact.parent.mkdir(parents=True)
    stale_artifact.write_bytes(b"stale")

    current_output_dir = tmp_path / "fresh-run"
    current_artifact = current_output_dir / "generated_model" / "generated_model_vela.tflite"
    current_artifact.parent.mkdir(parents=True)
    current_artifact.write_bytes(b"fresh")

    plan_path = write_execution_plan(
        repo_root=tmp_path,
        manifest_path=manifest,
        output_dir=current_output_dir,
        flash_capacity=0x00400000,
    )

    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    assert payload["batches"][0]["models"][0]["model_path"] == str(current_artifact)
    assert payload["batches"][0]["models"][0]["model_size_bytes"] == len(b"fresh")


def test_write_execution_plan_packs_batch_flash_addresses_by_real_size(tmp_path: Path) -> None:
    manifest = tmp_path / "models.yaml"
    manifest.write_text(
        """
models:
  - name: first_model
    task: classification
    source:
      type: torchvision
      model: mobilenet_v2
    input_shape: [1, 3, 224, 224]
    calibration_dataset: imagenet_1k_random
    tier: 1
    flash_address: 0x00A00000
  - name: second_model
    task: classification
    source:
      type: torchvision
      model: mobilenet_v3_small
    input_shape: [1, 3, 224, 224]
    calibration_dataset: imagenet_1k_random
    tier: 1
    flash_address: 0x00C00000
        """.strip(),
        encoding="utf-8",
    )
    output_dir = tmp_path / "packed-run"
    first_artifact = output_dir / "first_model" / "first_model_vela.tflite"
    second_artifact = output_dir / "second_model" / "second_model_vela.tflite"
    first_artifact.parent.mkdir(parents=True)
    second_artifact.parent.mkdir(parents=True)
    first_artifact.write_bytes(b"a" * 0x1800)
    second_artifact.write_bytes(b"b" * 0x1400)

    plan_path = write_execution_plan(
        repo_root=tmp_path,
        manifest_path=manifest,
        output_dir=output_dir,
        flash_capacity=0x00400000,
    )

    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    models = payload["batches"][0]["models"]
    assert models[0]["flash_address"] == "0x400000"
    assert models[1]["flash_address"] == "0x402000"


def test_write_execution_plan_can_add_model_zoo_baseline_variants(tmp_path: Path) -> None:
    manifest = tmp_path / "models.yaml"
    manifest.write_text(
        """
models:
  - name: yolo11n_od_192
    task: object_detection
    source:
      type: ultralytics
      model: yolo11n.pt
    input_shape: [1, 3, 192, 192]
    calibration_dataset: coco_128
    tier: 1
    flash_address: 0x00400000
    model_zoo_ref: yolo11n_ref.tflite
        """.strip(),
        encoding="utf-8",
    )
    output_dir = tmp_path / "run"
    artifact_path = output_dir / "yolo11n_od_192" / "yolo11n_od_192_vela.tflite"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(b"generated")
    model_zoo_path = tmp_path / "model_zoo" / "tflm_yolo11_od" / "yolo11n_ref.tflite"
    model_zoo_path.parent.mkdir(parents=True)
    model_zoo_path.write_bytes(b"reference")
    summary_path = output_dir / "conversion_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "name": "yolo11n_od_192",
                        "status": "exported_from_source",
                        "model_zoo_ref": "yolo11n_ref.tflite",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    plan_path = write_execution_plan(
        repo_root=tmp_path,
        manifest_path=manifest,
        output_dir=output_dir,
        flash_capacity=0x00400000,
        conversion_summary_path=summary_path,
        include_model_zoo_baselines=True,
    )

    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    models = payload["batches"][0]["models"]
    assert [model["name"] for model in models] == [
        "yolo11n_od_192 (ours)",
        "yolo11n_od_192 (model_zoo)",
    ]
    assert models[0]["model_path"] == str(artifact_path)
    assert models[1]["model_path"] == str(model_zoo_path)


def test_write_execution_plan_skips_model_zoo_baseline_when_source_export_not_real(tmp_path: Path) -> None:
    manifest = tmp_path / "models.yaml"
    manifest.write_text(
        """
models:
  - name: yolo11n_od_192
    task: object_detection
    source:
      type: ultralytics
      model: yolo11n.pt
    input_shape: [1, 3, 192, 192]
    calibration_dataset: coco_128
    tier: 1
    flash_address: 0x00400000
    model_zoo_ref: yolo11n_ref.tflite
        """.strip(),
        encoding="utf-8",
    )
    output_dir = tmp_path / "run"
    artifact_path = output_dir / "yolo11n_od_192" / "yolo11n_od_192_vela.tflite"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(b"generated")
    model_zoo_path = tmp_path / "model_zoo" / "tflm_yolo11_od" / "yolo11n_ref.tflite"
    model_zoo_path.parent.mkdir(parents=True)
    model_zoo_path.write_bytes(b"reference")
    summary_path = output_dir / "conversion_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "name": "yolo11n_od_192",
                        "status": "staged_model_zoo_ref",
                        "model_zoo_ref": "yolo11n_ref.tflite",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    plan_path = write_execution_plan(
        repo_root=tmp_path,
        manifest_path=manifest,
        output_dir=output_dir,
        flash_capacity=0x00400000,
        conversion_summary_path=summary_path,
        include_model_zoo_baselines=True,
    )

    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    models = payload["batches"][0]["models"]
    assert [model["name"] for model in models] == ["yolo11n_od_192"]


def test_prepare_artifacts_delegates_to_convert_manifest(tmp_path: Path) -> None:
    with (
        patch("tools.benchmark_runner.convert_manifest", return_value=["result"]) as convert_mock,
        patch("tools.benchmark_runner.write_conversion_summary", return_value=tmp_path / "summary.json") as summary_mock,
    ):
        summary_path = prepare_artifacts(
            repo_root=tmp_path,
            manifest_path=tmp_path / "models.yaml",
            output_dir=tmp_path / "artifacts",
            vela_dir=tmp_path / "artifacts" / "vela",
            skip_vela=True,
            allow_model_zoo_ref=False,
            isolate_source_exports=True,
            tiers={1, 2},
        )

    assert summary_path == tmp_path / "summary.json"
    convert_mock.assert_called_once()
    assert convert_mock.call_args.kwargs["isolate_source_exports"] is True
    assert convert_mock.call_args.kwargs["tiers"] == {1, 2}
    summary_mock.assert_called_once_with(["result"], tmp_path / "artifacts")


def test_run_execution_plan_flashes_batches_and_merges_board_results(tmp_path: Path) -> None:
    plan_path = tmp_path / "benchmark_plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "batches": [
                    {
                        "batch_index": 0,
                        "table_path": str(tmp_path / "model_table_batch_0.bin"),
                        "models": [
                            {
                                "name": "yolo11n_od_192",
                                "model_path": "artifacts/yolo11n_od_192/yolo11n_od_192_vela.tflite",
                                "flash_address": "0x400000",
                            }
                        ],
                    },
                    {
                        "batch_index": 1,
                        "table_path": str(tmp_path / "model_table_batch_1.bin"),
                        "models": [
                            {
                                "name": "yolov8n_od_192",
                                "model_path": "artifacts/yolov8n_od_192/yolov8n_od_192_vela.tflite",
                                "flash_address": "0x600000",
                            }
                        ],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    captured_payloads = [
        [
            {
                "benchmark": {"device": "himax_hx6538", "firmware": "model_benchmark_v1", "timestamp": 1},
                "models": [{"name": "yolo11n_od_192", "status": "ok"}],
            }
        ],
        [
            {
                "benchmark": {"device": "himax_hx6538", "firmware": "model_benchmark_v1", "timestamp": 2},
                "models": [{"name": "yolov8n_od_192", "status": "ok"}],
            }
        ],
    ]

    with (
        patch("tools.benchmark_runner.flash_model_batch") as flash_mock,
        patch("tools.benchmark_runner.capture_json_objects", side_effect=captured_payloads) as capture_mock,
    ):
        board_result_path = run_execution_plan(plan_path, port="COM7", output_dir=tmp_path)

    assert board_result_path == tmp_path / "benchmark_result_latest.json"
    payload = json.loads(board_result_path.read_text(encoding="utf-8"))
    assert [model["name"] for model in payload["models"]] == ["yolo11n_od_192", "yolov8n_od_192"]
    assert payload["benchmark"]["device"] == "himax_hx6538"
    assert flash_mock.call_count == 2
    assert capture_mock.call_count == 2
    assert flash_mock.call_args.kwargs["auto_reset"] is False


def test_run_execution_plan_writes_batch_logs(tmp_path: Path) -> None:
    plan_path = tmp_path / "benchmark_plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "batches": [
                    {
                        "batch_index": 3,
                        "table_path": str(tmp_path / "model_table_batch_3.bin"),
                        "models": [
                            {
                                "name": "yolo11n_od_192",
                                "model_path": "artifacts/yolo11n_od_192/yolo11n_od_192_vela.tflite",
                                "flash_address": "0x400000",
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with (
        patch("tools.benchmark_runner.flash_model_batch"),
        patch(
            "tools.benchmark_runner.capture_json_objects",
            return_value=[
                {
                    "benchmark": {"device": "himax_hx6538", "firmware": "model_benchmark_v1", "timestamp": 1},
                    "models": [{"name": "yolo11n_od_192", "status": "ok"}],
                }
            ],
        ) as capture_mock,
    ):
        run_execution_plan(plan_path, port="COM7", output_dir=tmp_path)

    assert capture_mock.call_args.kwargs["log_path"] == tmp_path / "serial_benchmark_capture_batch3.log"


def test_run_execution_plan_passes_model_chunk_size_to_flashing(tmp_path: Path) -> None:
    plan_path = tmp_path / "benchmark_plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "batches": [
                    {
                        "batch_index": 0,
                        "table_path": str(tmp_path / "model_table_batch_0.bin"),
                        "models": [
                            {
                                "name": "yolov8n_od_192",
                                "model_path": "artifacts/yolov8n_od_192/yolov8n_od_192_vela.tflite",
                                "flash_address": "0x600000",
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with (
        patch("tools.benchmark_runner.flash_model_batch") as flash_mock,
        patch(
            "tools.benchmark_runner.capture_json_objects",
            return_value=[
                {
                    "benchmark": {"device": "himax_hx6538", "firmware": "model_benchmark_v1", "timestamp": 1},
                    "models": [{"name": "yolov8n_od_192", "status": "ok"}],
                }
            ],
        ),
    ):
        run_execution_plan(
            plan_path,
            port="COM7",
            output_dir=tmp_path,
            flash_model_chunk_size=0x10000,
        )

    assert flash_mock.call_args.kwargs["model_chunk_size"] == 0x10000


def test_run_board_benchmark_can_flash_firmware_first(tmp_path: Path) -> None:
    plan_path = tmp_path / "benchmark_plan.json"
    plan_path.write_text(json.dumps({"batches": []}), encoding="utf-8")

    with (
        patch("tools.benchmark_runner.flash_firmware_image") as flash_firmware_mock,
        patch(
            "tools.benchmark_runner.run_execution_plan",
            return_value=tmp_path / "benchmark_result_latest.json",
        ) as run_plan_mock,
    ):
        result = run_board_benchmark(
            plan_path,
            port="COM7",
            output_dir=tmp_path,
            flash_firmware_first=True,
        )

    assert result == tmp_path / "benchmark_result_latest.json"
    flash_firmware_mock.assert_called_once()
    run_plan_mock.assert_called_once()


def test_run_board_benchmark_passes_auto_reset_to_flashing(tmp_path: Path) -> None:
    plan_path = tmp_path / "benchmark_plan.json"
    plan_path.write_text(json.dumps({"batches": []}), encoding="utf-8")

    with (
        patch("tools.benchmark_runner.flash_firmware_image") as flash_firmware_mock,
        patch(
            "tools.benchmark_runner.run_execution_plan",
            return_value=tmp_path / "benchmark_result_latest.json",
        ) as run_plan_mock,
    ):
        run_board_benchmark(
            plan_path,
            port="COM7",
            output_dir=tmp_path,
            flash_firmware_first=True,
            auto_reset=True,
        )

    assert flash_firmware_mock.call_args.kwargs["auto_reset"] is True
    assert run_plan_mock.call_args.kwargs["auto_reset"] is True


def test_run_board_benchmark_passes_model_chunk_size_to_execution_plan(tmp_path: Path) -> None:
    plan_path = tmp_path / "benchmark_plan.json"
    plan_path.write_text(json.dumps({"batches": []}), encoding="utf-8")

    with (
        patch("tools.benchmark_runner.flash_firmware_image"),
        patch(
            "tools.benchmark_runner.run_execution_plan",
            return_value=tmp_path / "benchmark_result_latest.json",
        ) as run_plan_mock,
    ):
        run_board_benchmark(
            plan_path,
            port="COM7",
            output_dir=tmp_path,
            flash_model_chunk_size=0x10000,
        )

    assert run_plan_mock.call_args.kwargs["flash_model_chunk_size"] == 0x10000


def test_parse_args_supports_from_board_result_and_skip_flash_aliases(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "benchmark_runner.py",
            "--from-board-result",
            "artifacts/board.json",
            "--skip-flash",
            "--isolate-source-exports",
        ],
    )

    args = parse_args()

    assert args.board_result == "artifacts/board.json"
    assert args.skip_flash is True
    assert args.isolate_source_exports is True


def test_parse_args_supports_auto_reset(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "benchmark_runner.py",
            "--port",
            "COM7",
            "--auto-reset",
        ],
    )

    args = parse_args()

    assert args.auto_reset is True


def test_parse_args_supports_flash_model_chunk_size(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "benchmark_runner.py",
            "--port",
            "COM7",
            "--flash-model-chunk-size",
            "0x10000",
        ],
    )

    args = parse_args()

    assert args.flash_model_chunk_size == "0x10000"


def test_parse_args_supports_tier_selection(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "benchmark_runner.py",
            "--tiers",
            "1,2",
        ],
    )

    args = parse_args()

    assert parse_tiers(args.tiers) == {1, 2}


def test_main_passes_conversion_summary_to_report(monkeypatch, tmp_path: Path) -> None:
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir()
    summary_path = output_dir / "conversion_summary.json"
    summary_path.write_text(json.dumps({"results": []}), encoding="utf-8")
    board_result = tmp_path / "board.json"
    board_result.write_text(json.dumps({"models": []}), encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "benchmark_runner.py",
            "--output-dir",
            str(output_dir),
            "--board-result",
            str(board_result),
            "--report-prefix",
            str(output_dir / "report"),
        ],
    )

    with (
        patch("tools.benchmark_runner.write_execution_plan", return_value=output_dir / "benchmark_plan.json"),
        patch("tools.benchmark_runner.load_payload", return_value={"models": []}),
        patch("tools.benchmark_runner.write_report", return_value=(output_dir / "report.md", output_dir / "report.csv")) as report_mock,
    ):
        assert main() == 0

    assert report_mock.call_args.kwargs["conversion_summary_path"] == summary_path


def test_capture_board_result_writes_latest_payload(tmp_path: Path) -> None:
    with patch(
        "tools.benchmark_runner.capture_json_objects",
        return_value=[
            {
                "benchmark": {"device": "himax_hx6538", "firmware": "model_benchmark_v1", "timestamp": 1},
                "models": [{"name": "yolo11n_od_192", "status": "ok"}],
            }
        ],
    ) as capture_mock:
        result = capture_board_result(port="COM7", output_dir=tmp_path)

    assert result == tmp_path / "benchmark_result_latest.json"
    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["models"][0]["name"] == "yolo11n_od_192"
    assert capture_mock.call_args.kwargs["log_path"] == tmp_path / "serial_benchmark_capture.log"


def test_run_board_benchmark_skip_flash_captures_only(tmp_path: Path) -> None:
    with (
        patch("tools.benchmark_runner.flash_firmware_image") as flash_firmware_mock,
        patch(
            "tools.benchmark_runner.capture_board_result",
            return_value=tmp_path / "benchmark_result_latest.json",
        ) as capture_mock,
        patch("tools.benchmark_runner.run_execution_plan") as run_plan_mock,
    ):
        result = run_board_benchmark(
            tmp_path / "benchmark_plan.json",
            port="COM7",
            output_dir=tmp_path,
            flash_firmware_first=False,
            skip_flash=True,
        )

    assert result == tmp_path / "benchmark_result_latest.json"
    flash_firmware_mock.assert_not_called()
    run_plan_mock.assert_not_called()
    capture_mock.assert_called_once()
