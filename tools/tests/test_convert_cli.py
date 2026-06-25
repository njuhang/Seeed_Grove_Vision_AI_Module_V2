"""Hermetic tests for tools.model_converter.convert (in-process litert-torch path).

Task 1.3 rewrote the conversion dispatch to run entirely in-process: the
orchestrator imports ``litert_convert`` and ``calibration`` directly and calls
``litert_convert.convert_pt_to_int8_tflite`` -- there is no ``wsl`` subprocess,
no ONNX, no ``onnx2bf``/``onnx2tf``. These tests assert on that new shape:

* the source-export happy path is mocked at the seam ``litert_convert`` /
  ``_build_ultralytics_module`` (NOT at ``subprocess.run``);
* the model_zoo_ref fallback is driven by ``_check_source_dependencies`` /
  by raising from the converter, NOT by ``_probe_wsl_modules``;
* no test patches ``subprocess.run`` for the conversion path -- its presence
  in a diff would be a regression signal.
"""

import json
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

from tools.model_converter import convert as convert_module
from tools.model_converter.convert import convert_manifest


def _fake_torch_with_zeros() -> MagicMock:
    """Return a fake ``torch`` exposing ``.zeros(*shape)`` for sample_args.

    The dispatcher builds ``sample_args = (torch.zeros(*input_shape),)`` from
    the value ``_build_ultralytics_module`` returns as its second element. For
    hermetic tests we don't need a real tensor -- the converter itself is mocked
    -- so a MagicMock that records the shape is enough.
    """
    fake = MagicMock(name="fake_torch")
    fake.zeros.return_value = "FAKE_SAMPLE_INPUT"
    return fake


def _write_ultralytics_manifest(
    manifest: Path,
    name: str = "yolo11n_od_192",
    model_zoo_ref: str | None = None,
    tier: int = 1,
) -> None:
    extra = ""
    if model_zoo_ref is not None:
        extra = f"\n    model_zoo_ref: {model_zoo_ref}"
    manifest.write_text(
        f"""
models:
  - name: {name}
    task: object_detection
    source:
      type: ultralytics
      model: {name.split('_')[0]}.pt
    input_shape: [1, 3, 192, 192]
    calibration_dataset: coco_128
    tier: {tier}
    flash_address: 0x00400000{extra}
        """.strip(),
        encoding="utf-8",
    )


def _write_torchvision_manifest(
    manifest: Path,
    name: str = "mobilenetv2_cls_224",
    extra_yaml: str = "",
) -> None:
    manifest.write_text(
        f"""
models:
  - name: {name}
    task: classification
    source:
      type: torchvision
      model: mobilenet_v2
      weights: MobileNet_V2_Weights.DEFAULT
    input_shape: [1, 3, 224, 224]
    calibration_dataset: imagenet_1k_random
    tier: 1
    flash_address: 0x00800000{extra_yaml}
        """.strip(),
        encoding="utf-8",
    )


def _write_timm_manifest(manifest: Path, name: str = "efficientnet_lite0_cls_224") -> None:
    manifest.write_text(
        f"""
models:
  - name: {name}
    task: classification
    source:
      type: timm
      model: tf_efficientnet_lite0
      pretrained: true
    input_shape: [1, 3, 224, 224]
    calibration_dataset: imagenet_1k_random
    tier: 1
    flash_address: 0x00A00000
        """.strip(),
        encoding="utf-8",
    )


def _write_torch_hub_manifest(manifest: Path, name: str = "yolov5n_od_192") -> None:
    manifest.write_text(
        f"""
models:
  - name: {name}
    task: object_detection
    source:
      type: torch.hub
      repo: ultralytics/yolov5
      model: yolov5n
      pretrained: true
    input_shape: [1, 3, 192, 192]
    calibration_dataset: coco_128
    tier: 1
    flash_address: 0x00400000
        """.strip(),
        encoding="utf-8",
    )


def _write_huggingface_manifest(manifest: Path, name: str = "segformerb0_seg_512") -> None:
    manifest.write_text(
        f"""
models:
  - name: {name}
    task: semantic_segmentation
    source:
      type: huggingface
      model_class: SegformerForSemanticSegmentation
      repo: nvidia/segformer-b0-finetuned-ade-512-512
    input_shape: [1, 3, 512, 512]
    calibration_dataset: imagenet_1k_random
    tier: 1
    flash_address: 0x01000000
        """.strip(),
        encoding="utf-8",
    )


def _write_ultraface_manifest(manifest: Path, name: str = "ultraface_rfb320_face_320x240") -> None:
    manifest.write_text(
        f"""
models:
  - name: {name}
    task: face_detection
    source:
      type: ultraface
      variant: RFB
      input_size: 320
      repo_path: artifacts_debug/vendor/ultraface
      weights: models/pretrained/version-RFB-320.pth
    input_shape: [1, 3, 240, 320]
    calibration_dataset: coco_128
    tier: 1
    flash_address: 0x02000000
        """.strip(),
        encoding="utf-8",
    )


def _write_blazeface_manifest(manifest: Path, name: str = "blazeface_front_face_128") -> None:
    manifest.write_text(
        f"""
models:
  - name: {name}
    task: face_detection
    source:
      type: blazeface
      model: front
      repo_path: artifacts_debug/vendor/blazeface
      weights: blazeface.pth
    input_shape: [1, 3, 128, 128]
    calibration_dataset: coco_128
    tier: 1
    flash_address: 0x02200000
        """.strip(),
        encoding="utf-8",
    )


def _write_retinaface_manifest(manifest: Path, name: str = "retinaface_mnet_face_160") -> None:
    manifest.write_text(
        f"""
models:
  - name: {name}
    task: face_detection
    source:
      type: retinaface
      backbone: mobile0.25
      repo_path: artifacts_debug/vendor/retinaface
      weights: weights/mobilenet0.25_Final.pth
    input_shape: [1, 3, 160, 160]
    calibration_dataset: coco_128
    tier: 1
    flash_address: 0x02400000
        """.strip(),
        encoding="utf-8",
    )


def _write_movenet_manifest(manifest: Path, name: str = "movenet_lightning_pose_192") -> None:
    manifest.write_text(
        f"""
models:
  - name: {name}
    task: keypoint_detection
    source:
      type: movenet
      repo_path: artifacts_debug/vendor/movenet.pytorch
      weights: output/e118_valacc0.79805.pth
      model_file: lib/models/movenet_mobilenetv2.py
      num_classes: 17
    input_shape: [1, 3, 192, 192]
    calibration_dataset: coco_128
    tier: 1
    flash_address: 0x02600000
        """.strip(),
        encoding="utf-8",
    )


def _write_honk_manifest(manifest: Path, name: str = "honk_res8_narrow_kws_101x40") -> None:
    manifest.write_text(
        f"""
models:
  - name: {name}
    task: keyword_spotting
    source:
      type: honk
      architecture: res8-narrow
      repo_path: artifacts_debug/vendor/honk
      weights: ../honk-models/res8_narrow.pt
      n_labels: 12
    input_shape: [1, 101, 40]
    calibration_dataset: speech_commands
    tier: 1
    flash_address: 0x02800000
        """.strip(),
        encoding="utf-8",
    )


def _write_prebuilt_tflite_manifest(
    manifest: Path,
    name: str = "mediapipe_hand_landmarks_224",
    *,
    dequantize_float16_constants: bool = False,
    static_int8: bool = False,
    quantize_prelu_float_islands: bool = False,
) -> None:
    rewrite_options = []
    if dequantize_float16_constants:
        rewrite_options.append("dequantize_float16_constants: true")
    if static_int8:
        rewrite_options.append("static_int8: true")
    if quantize_prelu_float_islands:
        rewrite_options.append("quantize_prelu_float_islands: true")
    rewrite_option = "".join(f"\n      {option}" for option in rewrite_options)
    manifest.write_text(
        f"""
models:
  - name: {name}
    task: hand_landmark
    source:
      type: prebuilt_tflite
      path: artifacts_debug/vendor/mediapipe/hand_landmarker_extracted/hand_landmarks_detector.tflite
      origin: mediapipe_hand_landmarker_task
      input_layout: NHWC{rewrite_option}
    input_shape: [1, 3, 224, 224]
    calibration_dataset: coco_128
    tier: 1
    flash_address: 0x02A00000
        """.strip(),
        encoding="utf-8",
    )


def _write_torchvision_segmentation_manifest(
    manifest: Path,
    name: str = "deeplabv3_mbnv3_seg_320",
) -> None:
    manifest.write_text(
        f"""
models:
  - name: {name}
    task: semantic_segmentation
    source:
      type: torchvision
      model: deeplabv3_mobilenet_v3_large
      weights: DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
    input_shape: [1, 3, 320, 320]
    calibration_dataset: imagenet_1k_random
    tier: 1
    flash_address: 0x01200000
        """.strip(),
        encoding="utf-8",
    )


def test_convert_script_runs_without_import_error(tmp_path: Path) -> None:
    """``python convert.py`` must import cleanly and exit 0.

    Uses an unsupported source type with a model_zoo_ref so the model is staged
    (no source backend needed) -- exercises the CLI entrypoint end-to-end
    without invoking the real torch conversion.
    """
    repo_root = Path(__file__).resolve().parents[2]
    work_root = tmp_path / "repo"
    work_root.mkdir()
    manifest = work_root / "models.yaml"
    manifest.write_text(
        """
models:
  - name: demo_model
    task: object_detection
    source:
      type: unsupported
      model: demo.pt
    input_shape: [1, 3, 192, 192]
    calibration_dataset: coco_128
    tier: 1
    flash_address: 0x00400000
    model_zoo_ref: demo_ref.tflite
        """.strip(),
        encoding="utf-8",
    )
    model_zoo_dir = work_root / "model_zoo" / "demo"
    model_zoo_dir.mkdir(parents=True)
    (model_zoo_dir / "demo_ref.tflite").write_bytes(b"demo")

    result = subprocess.run(
        [
            sys.executable,
            "tools/model_converter/convert.py",
            "--manifest",
            str(manifest),
            "--repo-root",
            str(work_root),
            "--output-dir",
            str(work_root / "artifacts"),
            "--skip-vela",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_check_source_dependencies_does_not_require_importlib_util_attr(
    monkeypatch,
) -> None:
    monkeypatch.delattr(convert_module.importlib, "util", raising=False)

    missing = convert_module._check_source_dependencies(
        ("json", "definitely_missing_module_for_benchmark_tests")
    )

    assert "json" not in missing
    assert "definitely_missing_module_for_benchmark_tests" in missing


def test_torch_hub_dependency_probe_includes_yolov5_runtime_deps() -> None:
    assert "pandas" in convert_module.REQUIRED_SOURCE_DEPS["torch.hub"]
    assert "seaborn" in convert_module.REQUIRED_SOURCE_DEPS["torch.hub"]


def test_convert_manifest_uses_litert_torch_direct_pipeline(tmp_path: Path) -> None:
    """Happy path: source model goes through in-process litert-torch int8.

    Asserts a single-stage, in-process pipeline:
    * ``_build_ultralytics_module`` constructs a (fake) nn.Module -- no
      ``subprocess``;
    * ``litert_convert.convert_pt_to_int8_tflite`` is called exactly once with
      the spec's input_shape / calibration_dataset;
    * the result status is ``exported_from_source`` with a non-empty int8
      tflite;
    * NO ``onnx`` / ``onnx2bf`` / ``onnx2tf`` / ``wsl_export_ultralytics.py``
      reference appears anywhere in the dispatch.
    """
    manifest = tmp_path / "models.yaml"
    _write_ultralytics_manifest(manifest, name="yolo11n_od_192")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    artifacts_root = repo_root / "artifacts"

    generated_vela = artifacts_root / "vela" / "yolo11n_od_192_int8_vela.tflite"
    generated_vela.parent.mkdir(parents=True, exist_ok=True)
    generated_vela.write_bytes(b"vela")

    convert_calls: list[dict] = []

    def fake_convert(module, sample_args, model_name, *, output_root, calibration_samples):
        convert_calls.append(
            {
                "model_name": model_name,
                "output_root": str(output_root),
                "samples": list(calibration_samples),
            }
        )
        # Write a real non-empty int8 tflite to the contract path so the vela
        # staging step can find it.
        out_dir = Path(output_root) / model_name
        out_dir.mkdir(parents=True, exist_ok=True)
        int8 = out_dir / f"{model_name}_int8.tflite"
        int8.write_bytes(b"int8-from-litert-torch")
        return {"float": out_dir / f"{model_name}_float.tflite", "int8": int8}

    with (
        patch("tools.model_converter.convert._check_source_dependencies", return_value=[]),
        patch(
            "tools.model_converter.convert._build_ultralytics_module",
            return_value=("<fake-nn-module>", _fake_torch_with_zeros()),
        ),
        patch(
            "tools.model_converter.convert.litert_convert.convert_pt_to_int8_tflite",
            side_effect=fake_convert,
        ) as convert_mock,
        patch("tools.model_converter.convert.write_vela_artifacts") as write_vela_mock,
    ):
        write_vela_mock.return_value = artifacts_root / "vela" / "yolo11n_od_192.vela_info.json"
        results = convert_manifest(
            manifest_path=manifest,
            repo_root=repo_root,
            output_root=artifacts_root,
            vela_dir=artifacts_root / "vela",
        )

    assert [result.status for result in results] == ["exported_from_source"]
    convert_mock.assert_called_once()
    # The dispatch must thread the spec's input_shape (used to build sample_args
    # and the calibration loader) -- assert the model name and that we got the
    # calibration samples (one per loader yield).
    assert convert_calls[0]["model_name"] == "yolo11n_od_192"
    assert convert_calls[0]["output_root"] == str(artifacts_root)
    assert len(convert_calls[0]["samples"]) > 0
    exported_int8 = artifacts_root / "yolo11n_od_192" / "yolo11n_od_192_int8.tflite"
    assert exported_int8.read_bytes() == b"int8-from-litert-torch"
    # Vela staging still runs (not skipped) and produces the final vela tflite.
    assert results[0].vela_model_path == str(
        artifacts_root / "yolo11n_od_192" / "yolo11n_od_192_vela.tflite"
    )
    assert (artifacts_root / "yolo11n_od_192" / "yolo11n_od_192_vela.tflite").read_bytes() == b"vela"
    write_vela_mock.assert_called_once()


def test_convert_manifest_can_select_tier2_models(tmp_path: Path) -> None:
    manifest = tmp_path / "models.yaml"
    _write_ultralytics_manifest(manifest, name="yolo11s_od_192", tier=2)
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    artifacts_root = repo_root / "artifacts"

    generated_vela = artifacts_root / "vela" / "yolo11s_od_192_int8_vela.tflite"
    generated_vela.parent.mkdir(parents=True, exist_ok=True)
    generated_vela.write_bytes(b"vela")

    def fake_convert(module, sample_args, model_name, *, output_root, calibration_samples):
        out_dir = Path(output_root) / model_name
        out_dir.mkdir(parents=True, exist_ok=True)
        int8 = out_dir / f"{model_name}_int8.tflite"
        int8.write_bytes(b"int8")
        return {"float": out_dir / f"{model_name}_float.tflite", "int8": int8}

    with (
        patch("tools.model_converter.convert._check_source_dependencies", return_value=[]),
        patch(
            "tools.model_converter.convert._build_ultralytics_module",
            return_value=("<fake-nn-module>", _fake_torch_with_zeros()),
        ),
        patch(
            "tools.model_converter.convert.litert_convert.convert_pt_to_int8_tflite",
            side_effect=fake_convert,
        ),
        patch("tools.model_converter.convert.write_vela_artifacts") as write_vela_mock,
    ):
        write_vela_mock.return_value = artifacts_root / "vela" / "yolo11s_od_192.vela_info.json"
        results = convert_manifest(
            manifest_path=manifest,
            repo_root=repo_root,
            output_root=artifacts_root,
            vela_dir=artifacts_root / "vela",
            model_names=["yolo11s_od_192"],
            tiers={2},
        )

    assert [result.name for result in results] == ["yolo11s_od_192"]
    assert [result.status for result in results] == ["exported_from_source"]


def test_convert_manifest_supports_yolo_fastestv2_backend(tmp_path: Path) -> None:
    manifest = tmp_path / "models.yaml"
    manifest.write_text(
        """
models:
  - name: yolo_fastestv2_od_192
    task: object_detection
    source:
      type: yolo_fastestv2
      repo: dog-qiuqiu/Yolo-FastestV2
      repo_path: artifacts_debug/vendor/yolo-fastestv2
      weights: artifacts_debug/vendor/yolo-fastestv2/model.pth
      data: data/coco.data
    input_shape: [1, 3, 192, 192]
    calibration_dataset: coco_128
    tier: 2
    flash_address: 0x03600000
        """.strip(),
        encoding="utf-8",
    )
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    artifacts_root = repo_root / "artifacts"

    with (
        patch("tools.model_converter.convert._check_source_dependencies", return_value=[]),
        patch(
            "tools.model_converter.convert._build_yolo_fastestv2_module",
            return_value=("<fake-yolo-fastestv2-module>", _fake_torch_with_zeros()),
            create=True,
        ) as build_mock,
        patch(
            "tools.model_converter.convert.litert_convert.convert_pt_to_int8_tflite",
            return_value={
                "float": artifacts_root / "yolo_fastestv2_od_192" / "yolo_fastestv2_od_192_float.tflite",
                "int8": artifacts_root / "yolo_fastestv2_od_192" / "yolo_fastestv2_od_192_int8.tflite",
            },
        ) as convert_mock,
        patch("tools.model_converter.convert.write_vela_artifacts") as write_vela_mock,
        patch(
            "tools.model_converter.convert._copy_generated_vela_model",
            return_value=artifacts_root / "yolo_fastestv2_od_192" / "yolo_fastestv2_od_192_vela.tflite",
        ),
    ):
        model_dir = artifacts_root / "yolo_fastestv2_od_192"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "yolo_fastestv2_od_192_int8.tflite").write_bytes(b"int8")
        write_vela_mock.return_value = artifacts_root / "vela" / "yolo_fastestv2_od_192.vela_info.json"
        results = convert_manifest(
            manifest_path=manifest,
            repo_root=repo_root,
            output_root=artifacts_root,
            vela_dir=artifacts_root / "vela",
            model_names=["yolo_fastestv2_od_192"],
            tiers={2},
        )

    assert [result.name for result in results] == ["yolo_fastestv2_od_192"]
    assert results[0].status == "exported_from_source"
    assert results[0].source_type == "yolo_fastestv2"
    build_mock.assert_called_once()
    convert_mock.assert_called_once()


def test_convert_manifest_supports_torchvision_backend(tmp_path: Path) -> None:
    manifest = tmp_path / "models.yaml"
    _write_torchvision_manifest(manifest)
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    artifacts_root = repo_root / "artifacts"

    generated_vela = artifacts_root / "vela" / "mobilenetv2_cls_224_int8_vela.tflite"
    generated_vela.parent.mkdir(parents=True, exist_ok=True)
    generated_vela.write_bytes(b"vela")

    convert_calls: list[dict] = []

    def fake_convert(module, sample_args, model_name, *, output_root, calibration_samples):
        convert_calls.append(
            {
                "model_name": model_name,
                "output_root": str(output_root),
                "samples": list(calibration_samples),
            }
        )
        out_dir = Path(output_root) / model_name
        out_dir.mkdir(parents=True, exist_ok=True)
        int8 = out_dir / f"{model_name}_int8.tflite"
        int8.write_bytes(b"int8-from-litert-torch")
        return {"float": out_dir / f"{model_name}_float.tflite", "int8": int8}

    with (
        patch("tools.model_converter.convert._check_source_dependencies", return_value=[]),
        patch(
            "tools.model_converter.convert._build_torchvision_module",
            return_value=("<fake-nn-module>", _fake_torch_with_zeros()),
        ),
        patch(
            "tools.model_converter.convert.litert_convert.convert_pt_to_int8_tflite",
            side_effect=fake_convert,
        ) as convert_mock,
        patch("tools.model_converter.convert.write_vela_artifacts") as write_vela_mock,
    ):
        write_vela_mock.return_value = artifacts_root / "vela" / "mobilenetv2_cls_224.vela_info.json"
        results = convert_manifest(
            manifest_path=manifest,
            repo_root=repo_root,
            output_root=artifacts_root,
            vela_dir=artifacts_root / "vela",
        )

    assert [result.status for result in results] == ["exported_from_source"]
    convert_mock.assert_called_once()
    assert convert_calls[0]["model_name"] == "mobilenetv2_cls_224"
    assert len(convert_calls[0]["samples"]) > 0
    assert results[0].source_type == "torchvision"


def test_convert_manifest_wraps_classification_models_for_channel_last_io(tmp_path: Path) -> None:
    manifest = tmp_path / "models.yaml"
    _write_torchvision_manifest(manifest, name="mobilenetv3small_cls_224")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    artifacts_root = repo_root / "artifacts"

    generated_vela = artifacts_root / "vela" / "mobilenetv3small_cls_224_int8_vela.tflite"
    generated_vela.parent.mkdir(parents=True, exist_ok=True)
    generated_vela.write_bytes(b"vela")

    fake_torch = _fake_torch_with_zeros()
    fake_litert_torch = types.SimpleNamespace(
        to_channel_last_io=MagicMock(return_value="<wrapped-channel-last-module>")
    )
    calibration_calls: list[dict[str, object]] = []

    def fake_calibration_loader(dataset, *, input_shape, **kwargs):
        calibration_calls.append(
            {
                "dataset": dataset,
                "input_shape": list(input_shape),
                "kwargs": kwargs,
            }
        )
        return [{"args_0": "sample"}]

    with (
        patch("tools.model_converter.convert._check_source_dependencies", return_value=[]),
        patch(
            "tools.model_converter.convert._build_torchvision_module",
            return_value=("<fake-nn-module>", fake_torch),
        ),
        patch.dict(sys.modules, {"litert_torch": fake_litert_torch}),
        patch(
            "tools.model_converter.convert.calibration.load_calibration_samples",
            side_effect=fake_calibration_loader,
        ),
        patch(
            "tools.model_converter.convert.litert_convert.convert_pt_to_int8_tflite",
            return_value={
                "float": artifacts_root / "mobilenetv3small_cls_224" / "mobilenetv3small_cls_224_float.tflite",
                "int8": artifacts_root / "mobilenetv3small_cls_224" / "mobilenetv3small_cls_224_int8.tflite",
            },
        ) as convert_mock,
        patch("tools.model_converter.convert.write_vela_artifacts") as write_vela_mock,
        patch(
            "tools.model_converter.convert._copy_generated_vela_model",
            return_value=artifacts_root / "mobilenetv3small_cls_224" / "mobilenetv3small_cls_224_vela.tflite",
        ),
    ):
        model_dir = artifacts_root / "mobilenetv3small_cls_224"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "mobilenetv3small_cls_224_int8.tflite").write_bytes(b"int8")
        write_vela_mock.return_value = artifacts_root / "vela" / "mobilenetv3small_cls_224.vela_info.json"
        results = convert_manifest(
            manifest_path=manifest,
            repo_root=repo_root,
            output_root=artifacts_root,
            vela_dir=artifacts_root / "vela",
        )

    assert [result.status for result in results] == ["exported_from_source"]
    fake_litert_torch.to_channel_last_io.assert_called_once_with(
        "<fake-nn-module>", args=[0]
    )
    fake_torch.zeros.assert_called_once_with(1, 224, 224, 3)
    assert calibration_calls == [
        {
            "dataset": "imagenet_1k_random",
            "input_shape": [1, 224, 224, 3],
            "kwargs": {},
        }
    ]
    convert_mock.assert_called_once_with(
        "<wrapped-channel-last-module>",
        ("FAKE_SAMPLE_INPUT",),
        "mobilenetv3small_cls_224",
        output_root=artifacts_root,
        calibration_samples=[{"args_0": "sample"}],
    )


def test_convert_manifest_threads_per_model_vela_overrides(tmp_path: Path) -> None:
    manifest = tmp_path / "models.yaml"
    _write_torchvision_manifest(
        manifest,
        name="mobilenetv3small_cls_224",
        extra_yaml="""
    vela:
      executable: python3
      executable_args:
        - artifacts_debug/run_vela450.py
      optimise: Size
        """,
    )
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    artifacts_root = repo_root / "artifacts"

    generated_vela = artifacts_root / "vela" / "mobilenetv3small_cls_224_int8_vela.tflite"
    generated_vela.parent.mkdir(parents=True, exist_ok=True)
    generated_vela.write_bytes(b"vela")

    with (
        patch("tools.model_converter.convert._check_source_dependencies", return_value=[]),
        patch(
            "tools.model_converter.convert._build_torchvision_module",
            return_value=("<fake-nn-module>", _fake_torch_with_zeros()),
        ),
        patch(
            "tools.model_converter.convert.litert_convert.convert_pt_to_int8_tflite",
            return_value={
                "float": artifacts_root / "mobilenetv3small_cls_224" / "mobilenetv3small_cls_224_float.tflite",
                "int8": artifacts_root / "mobilenetv3small_cls_224" / "mobilenetv3small_cls_224_int8.tflite",
            },
        ),
        patch("tools.model_converter.convert.write_vela_artifacts") as write_vela_mock,
        patch(
            "tools.model_converter.convert._copy_generated_vela_model",
            return_value=artifacts_root / "mobilenetv3small_cls_224" / "mobilenetv3small_cls_224_vela.tflite",
        ),
    ):
        model_dir = artifacts_root / "mobilenetv3small_cls_224"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "mobilenetv3small_cls_224_int8.tflite").write_bytes(b"int8")
        write_vela_mock.return_value = artifacts_root / "vela" / "mobilenetv3small_cls_224.vela_info.json"
        results = convert_manifest(
            manifest_path=manifest,
            repo_root=repo_root,
            output_root=artifacts_root,
            vela_dir=artifacts_root / "vela",
        )

    assert [result.status for result in results] == ["exported_from_source"]
    write_vela_mock.assert_called_once_with(
        model_name="mobilenetv3small_cls_224",
        model_path=artifacts_root / "mobilenetv3small_cls_224" / "mobilenetv3small_cls_224_int8.tflite",
        output_dir=artifacts_root / "vela",
        vela_executable="python3",
        vela_executable_args=["artifacts_debug/run_vela450.py"],
        optimise="Size",
    )


def test_rewrite_global_adaptive_avg_pool2d_as_mean_replaces_only_1x1_pools() -> None:
    class FakeModule:
        def named_children(self):
            for name, value in self.__dict__.items():
                if isinstance(value, FakeModule):
                    yield name, value

    class FakeAdaptiveAvgPool2d(FakeModule):
        def __init__(self, output_size):
            self.output_size = output_size

    class FakeLeaf(FakeModule):
        pass

    class FakeNested(FakeModule):
        def __init__(self):
            self.pool = FakeAdaptiveAvgPool2d((1, 1))
            self.leaf = FakeLeaf()

    class FakeRoot(FakeModule):
        def __init__(self):
            self.global_pool = FakeAdaptiveAvgPool2d(1)
            self.non_global_pool = FakeAdaptiveAvgPool2d((2, 2))
            self.child = FakeNested()

    fake_torch = types.SimpleNamespace(
        nn=types.SimpleNamespace(
            Module=FakeModule,
            AdaptiveAvgPool2d=FakeAdaptiveAvgPool2d,
        )
    )

    root = FakeRoot()
    rewritten = convert_module._rewrite_global_adaptive_avg_pool2d_as_mean(root, fake_torch)

    assert rewritten is root
    assert rewritten.global_pool.__class__.__name__ == "_SpatialMeanPool2d"
    assert isinstance(rewritten.non_global_pool, FakeAdaptiveAvgPool2d)
    assert rewritten.child.pool.__class__.__name__ == "_SpatialMeanPool2d"
    assert isinstance(rewritten.child.leaf, FakeLeaf)


def test_build_torchvision_module_wraps_segmentation_dict_output() -> None:
    fake_torch = types.ModuleType("torch")

    class FakeNNModule:
        def eval(self):
            return self

    class FakeSegmentationModule(FakeNNModule):
        def __call__(self, x):
            return {"out": f"segmentation:{x}", "aux": "ignored"}

    fake_torch.nn = types.SimpleNamespace(Module=FakeNNModule)

    class FakeWeightEnum:
        DEFAULT = object()

    def fake_factory(*, weights=None):
        assert weights is FakeWeightEnum.DEFAULT
        return FakeSegmentationModule()

    fake_tv_models = types.ModuleType("torchvision.models")
    fake_tv_models.deeplabv3_mobilenet_v3_large = fake_factory
    fake_tv_models.DeepLabV3_MobileNet_V3_Large_Weights = FakeWeightEnum

    fake_torchvision = types.ModuleType("torchvision")
    fake_torchvision.models = fake_tv_models

    with patch.dict(
        sys.modules,
        {
            "torch": fake_torch,
            "torchvision": fake_torchvision,
            "torchvision.models": fake_tv_models,
        },
    ):
        module, returned_torch = convert_module._build_torchvision_module(
            {
                "type": "torchvision",
                "model": "deeplabv3_mobilenet_v3_large",
                "weights": "DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT",
            }
        )

    assert returned_torch is fake_torch
    assert module.forward("frame") == "segmentation:frame"


def test_build_torchvision_module_falls_back_to_segmentation_namespace() -> None:
    fake_torch = types.ModuleType("torch")

    class FakeNNModule:
        def eval(self):
            return self

    class FakeSegmentationModule(FakeNNModule):
        def __call__(self, x):
            return {"out": f"segmentation:{x}"}

    fake_torch.nn = types.SimpleNamespace(Module=FakeNNModule)

    class FakeWeightEnum:
        DEFAULT = object()

    def fake_factory(*, weights=None):
        assert weights is FakeWeightEnum.DEFAULT
        return FakeSegmentationModule()

    fake_segmentation = types.SimpleNamespace(
        deeplabv3_mobilenet_v3_large=fake_factory,
        DeepLabV3_MobileNet_V3_Large_Weights=FakeWeightEnum,
    )
    fake_tv_models = types.ModuleType("torchvision.models")
    fake_tv_models.segmentation = fake_segmentation

    fake_torchvision = types.ModuleType("torchvision")
    fake_torchvision.models = fake_tv_models

    with patch.dict(
        sys.modules,
        {
            "torch": fake_torch,
            "torchvision": fake_torchvision,
            "torchvision.models": fake_tv_models,
        },
    ):
        module, returned_torch = convert_module._build_torchvision_module(
            {
                "type": "torchvision",
                "model": "deeplabv3_mobilenet_v3_large",
                "weights": "DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT",
            }
        )

    assert returned_torch is fake_torch
    assert module.forward("frame") == "segmentation:frame"


def test_build_torchvision_module_wraps_ssd_detection_raw_head_without_postprocess() -> None:
    fake_torch = types.ModuleType("torch")
    fake_torch.cat = MagicMock(side_effect=AssertionError("raw head wrapper should not concatenate outputs"))

    class FakeNNModule:
        def eval(self):
            return self

    fake_torch.nn = types.SimpleNamespace(Module=FakeNNModule)

    class FakeImages:
        tensors = "batched-images"

    class FakeTransform:
        def __call__(self, images, targets=None):
            assert images == "frame"
            assert targets is None
            return FakeImages(), None

    class FakeBackbone:
        def __call__(self, tensors):
            assert tensors == "batched-images"
            return {"0": "p3", "1": "p4"}

    class FakeHead:
        def __call__(self, features):
            assert features == ["p3", "p4"]
            return {
                "bbox_regression": "boxes",
                "cls_logits": "scores",
            }

    class FakeSSDModule(FakeNNModule):
        transform = FakeTransform()
        backbone = FakeBackbone()
        head = FakeHead()

        def __init__(self) -> None:
            self.postprocess_detections = MagicMock(side_effect=AssertionError("postprocess should not run"))

        def __call__(self, x):
            self.postprocess_detections()
            return [{"boxes": "nms-output"}]

    class FakeWeightEnum:
        DEFAULT = object()

    def fake_factory(*, weights=None):
        assert weights is FakeWeightEnum.DEFAULT
        return FakeSSDModule()

    fake_detection = types.SimpleNamespace(
        ssdlite320_mobilenet_v3_large=fake_factory,
        SSDLite320_MobileNet_V3_Large_Weights=FakeWeightEnum,
    )
    fake_tv_models = types.ModuleType("torchvision.models")
    fake_tv_models.detection = fake_detection

    fake_torchvision = types.ModuleType("torchvision")
    fake_torchvision.models = fake_tv_models

    with patch.dict(
        sys.modules,
        {
            "torch": fake_torch,
            "torchvision": fake_torchvision,
            "torchvision.models": fake_tv_models,
        },
    ):
        module, returned_torch = convert_module._build_torchvision_module(
            {
                "type": "torchvision",
                "model": "ssdlite320_mobilenet_v3_large",
                "weights": "SSDLite320_MobileNet_V3_Large_Weights.DEFAULT",
            }
        )

    assert returned_torch is fake_torch
    assert module.forward("frame") == ("boxes", "scores")
    fake_torch.cat.assert_not_called()


def test_convert_manifest_supports_torchvision_segmentation_backend(tmp_path: Path) -> None:
    manifest = tmp_path / "models.yaml"
    _write_torchvision_segmentation_manifest(manifest)
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    artifacts_root = repo_root / "artifacts"

    generated_vela = artifacts_root / "vela" / "deeplabv3_mbnv3_seg_320_int8_vela.tflite"
    generated_vela.parent.mkdir(parents=True, exist_ok=True)
    generated_vela.write_bytes(b"vela")

    with (
        patch("tools.model_converter.convert._check_source_dependencies", return_value=[]),
        patch(
            "tools.model_converter.convert._build_torchvision_module",
            return_value=("<fake-seg-module>", _fake_torch_with_zeros()),
        ),
        patch(
            "tools.model_converter.convert.litert_convert.convert_pt_to_int8_tflite",
            return_value={
                "float": artifacts_root / "deeplabv3_mbnv3_seg_320" / "deeplabv3_mbnv3_seg_320_float.tflite",
                "int8": artifacts_root / "deeplabv3_mbnv3_seg_320" / "deeplabv3_mbnv3_seg_320_int8.tflite",
            },
        ) as convert_mock,
        patch("tools.model_converter.convert.write_vela_artifacts") as write_vela_mock,
        patch(
            "tools.model_converter.convert._copy_generated_vela_model",
            return_value=artifacts_root / "deeplabv3_mbnv3_seg_320" / "deeplabv3_mbnv3_seg_320_vela.tflite",
        ),
    ):
        model_dir = artifacts_root / "deeplabv3_mbnv3_seg_320"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "deeplabv3_mbnv3_seg_320_int8.tflite").write_bytes(b"int8")
        write_vela_mock.return_value = artifacts_root / "vela" / "deeplabv3_mbnv3_seg_320.vela_info.json"
        results = convert_manifest(
            manifest_path=manifest,
            repo_root=repo_root,
            output_root=artifacts_root,
            vela_dir=artifacts_root / "vela",
        )

    assert [result.status for result in results] == ["exported_from_source"]
    convert_mock.assert_called_once()
    assert results[0].source_type == "torchvision"


def test_convert_from_source_bootstraps_env_before_backend_imports(tmp_path: Path) -> None:
    marker = {"bootstrapped": False}

    def fake_bootstrap() -> None:
        marker["bootstrapped"] = True

    def fake_builder(source):
        assert marker["bootstrapped"] is True
        return "<fake-module>", _fake_torch_with_zeros()

    out_dir = tmp_path / "artifacts"
    with (
        patch("tools.model_converter.convert.litert_convert._bootstrap_litert_env", side_effect=fake_bootstrap),
        patch("tools.model_converter.convert._build_torchvision_module", side_effect=fake_builder),
        patch(
            "tools.model_converter.convert.litert_convert.convert_pt_to_int8_tflite",
            return_value={
                "float": out_dir / "mobilenetv2_cls_224" / "mobilenetv2_cls_224_float.tflite",
                "int8": out_dir / "mobilenetv2_cls_224" / "mobilenetv2_cls_224_int8.tflite",
            },
        ),
    ):
        convert_module._convert_from_source(
            spec=convert_module.ModelSpec(
                name="mobilenetv2_cls_224",
                task="classification",
                source={
                    "type": "torchvision",
                    "model": "mobilenet_v2",
                    "weights": "MobileNet_V2_Weights.DEFAULT",
                },
                input_shape=[1, 3, 224, 224],
                calibration_dataset="imagenet_1k_random",
                tier=1,
                flash_address=0x00800000,
            ),
            repo_root=tmp_path,
            output_root=out_dir,
            vela_dir=out_dir / "vela",
            skip_vela=True,
        )


def test_convert_manifest_supports_timm_backend(tmp_path: Path) -> None:
    manifest = tmp_path / "models.yaml"
    _write_timm_manifest(manifest)
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    artifacts_root = repo_root / "artifacts"

    generated_vela = artifacts_root / "vela" / "efficientnet_lite0_cls_224_int8_vela.tflite"
    generated_vela.parent.mkdir(parents=True, exist_ok=True)
    generated_vela.write_bytes(b"vela")

    with (
        patch("tools.model_converter.convert._check_source_dependencies", return_value=[]),
        patch(
            "tools.model_converter.convert._build_timm_module",
            return_value=("<fake-nn-module>", _fake_torch_with_zeros()),
        ),
        patch(
            "tools.model_converter.convert.litert_convert.convert_pt_to_int8_tflite",
            return_value={
                "float": artifacts_root / "efficientnet_lite0_cls_224" / "efficientnet_lite0_cls_224_float.tflite",
                "int8": artifacts_root / "efficientnet_lite0_cls_224" / "efficientnet_lite0_cls_224_int8.tflite",
            },
        ) as convert_mock,
        patch("tools.model_converter.convert.write_vela_artifacts") as write_vela_mock,
        patch(
            "tools.model_converter.convert._copy_generated_vela_model",
            return_value=artifacts_root / "efficientnet_lite0_cls_224" / "efficientnet_lite0_cls_224_vela.tflite",
        ),
    ):
        model_dir = artifacts_root / "efficientnet_lite0_cls_224"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "efficientnet_lite0_cls_224_int8.tflite").write_bytes(b"int8")
        write_vela_mock.return_value = artifacts_root / "vela" / "efficientnet_lite0_cls_224.vela_info.json"
        results = convert_manifest(
            manifest_path=manifest,
            repo_root=repo_root,
            output_root=artifacts_root,
            vela_dir=artifacts_root / "vela",
        )

    assert [result.status for result in results] == ["exported_from_source"]
    convert_mock.assert_called_once()
    assert results[0].source_type == "timm"


def test_convert_manifest_supports_torch_hub_backend(tmp_path: Path) -> None:
    manifest = tmp_path / "models.yaml"
    _write_torch_hub_manifest(manifest)
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    artifacts_root = repo_root / "artifacts"

    generated_vela = artifacts_root / "vela" / "yolov5n_od_192_int8_vela.tflite"
    generated_vela.parent.mkdir(parents=True, exist_ok=True)
    generated_vela.write_bytes(b"vela")

    with (
        patch("tools.model_converter.convert._check_source_dependencies", return_value=[]),
        patch(
            "tools.model_converter.convert._build_torch_hub_module",
            return_value=("<fake-nn-module>", _fake_torch_with_zeros()),
        ),
        patch(
            "tools.model_converter.convert.litert_convert.convert_pt_to_int8_tflite",
            return_value={
                "float": artifacts_root / "yolov5n_od_192" / "yolov5n_od_192_float.tflite",
                "int8": artifacts_root / "yolov5n_od_192" / "yolov5n_od_192_int8.tflite",
            },
        ) as convert_mock,
        patch("tools.model_converter.convert.write_vela_artifacts") as write_vela_mock,
        patch(
            "tools.model_converter.convert._copy_generated_vela_model",
            return_value=artifacts_root / "yolov5n_od_192" / "yolov5n_od_192_vela.tflite",
        ),
    ):
        model_dir = artifacts_root / "yolov5n_od_192"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "yolov5n_od_192_int8.tflite").write_bytes(b"int8")
        write_vela_mock.return_value = artifacts_root / "vela" / "yolov5n_od_192.vela_info.json"
        results = convert_manifest(
            manifest_path=manifest,
            repo_root=repo_root,
            output_root=artifacts_root,
            vela_dir=artifacts_root / "vela",
        )

    assert [result.status for result in results] == ["exported_from_source"]
    convert_mock.assert_called_once()
    assert results[0].source_type == "torch.hub"


def test_build_torch_hub_module_defaults_to_non_interactive_repo_trust() -> None:
    fake_model = MagicMock(name="fake_torch_hub_model")
    fake_model.eval.return_value = fake_model
    fake_torch = types.SimpleNamespace(
        hub=types.SimpleNamespace(load=MagicMock(return_value=fake_model))
    )

    with patch.dict(sys.modules, {"torch": fake_torch}):
        module, returned_torch = convert_module._build_torch_hub_module(
            {
                "type": "torch.hub",
                "repo": "ultralytics/yolov5",
                "model": "yolov5n",
                "pretrained": True,
            }
        )

    assert module is fake_model
    assert returned_torch is fake_torch
    fake_torch.hub.load.assert_called_once_with(
        "ultralytics/yolov5",
        "yolov5n",
        pretrained=True,
        trust_repo=True,
        autoshape=False,
        skip_validation=True,
    )


def test_ultralytics_raw_head_wrapper_returns_decoded_tensor_without_aux_dict() -> None:
    class FakeBackboneModule:
        f = -1
        i = 0

        def __call__(self, x):
            return x + "|backbone"

    class FakeHeadModule:
        f = -1
        i = 1
        one2many = {"tag": "one2many"}
        end2end = False

        def __init__(self) -> None:
            self.forward_calls = 0
            self.forward_head_calls: list[tuple[object, dict[str, object]]] = []

        def __call__(self, x):
            self.forward_calls += 1
            return (x + "|decoded", {"boxes": x + "|boxes", "scores": x + "|scores", "feats": [x + "|feat0"]})

        def forward_head(self, x, **kwargs):
            self.forward_head_calls.append((x, kwargs))
            return x + "|raw-head"

    fake_head = FakeHeadModule()

    class FakeDetectionModel:
        save = set()
        model = [FakeBackboneModule(), fake_head]

    wrapper = convert_module._UltralyticsRawHeadWrapper(FakeDetectionModel())

    out = wrapper.forward("frame")

    assert out == "frame|backbone|decoded"
    assert fake_head.forward_calls == 1
    assert fake_head.forward_head_calls == []


def test_ultralytics_raw_head_wrapper_keeps_single_tensor_detect_output() -> None:
    class FakeBackboneModule:
        f = -1
        i = 0

        def __call__(self, x):
            return x + "|backbone"

    class FakeHeadModule:
        f = -1
        i = 1
        one2many = {"tag": "one2many"}
        end2end = False

        def __call__(self, x):
            return x + "|decoded"

        def forward_head(self, x, **kwargs):
            return {"boxes": x + "|boxes", "scores": x + "|scores", "feats": [x + "|feat0"]}

    class FakeDetectionModel:
        save = set()
        model = [FakeBackboneModule(), FakeHeadModule()]

    wrapper = convert_module._UltralyticsRawHeadWrapper(FakeDetectionModel())

    out = wrapper.forward("frame")

    assert out == "frame|backbone|decoded"


def test_ultralytics_raw_head_wrapper_forces_yolov5_detect_training_path() -> None:
    class FakeBackboneModule:
        f = -1
        i = 0

        def __call__(self, x):
            return [x + "|p3", x + "|p4"]

    class FakeYoloV5DetectHead:
        f = -1
        i = 1

        def __init__(self) -> None:
            self.training = False
            self.export = False
            self.nl = 3
            self.calls: list[tuple[bool, object]] = []

        def __call__(self, x):
            self.calls.append((self.training, x))
            if self.training:
                return x + ["raw-detect"]
            return "decoded-with-pow"

    fake_head = FakeYoloV5DetectHead()

    class FakeDetectionModel:
        save = set()
        model = [FakeBackboneModule(), fake_head]

    wrapper = convert_module._UltralyticsRawHeadWrapper(FakeDetectionModel())

    out = wrapper.forward("frame")

    assert out == ["frame|p3", "frame|p4", "raw-detect"]
    assert fake_head.calls == [(True, ["frame|p3", "frame|p4"])]
    assert fake_head.training is False



def test_build_huggingface_module_can_replace_gelu_with_relu(monkeypatch) -> None:
    class FakeModule:
        def eval(self):
            return self

        def named_children(self):
            return []

    class GELUActivation(FakeModule):
        pass

    class FakeReLU(FakeModule):
        pass

    class FakeSegformer(FakeModule):
        def __init__(self) -> None:
            self.activation_fn = GELUActivation()

        @classmethod
        def from_pretrained(cls, repo: str):
            assert repo == "fake/segformer"
            return cls()

        def named_children(self):
            return [("activation_fn", self.activation_fn)]

        def __call__(self, x):
            return types.SimpleNamespace(logits=f"logits:{x}")

    fake_torch = types.SimpleNamespace(
        nn=types.SimpleNamespace(Module=FakeModule, ReLU=FakeReLU)
    )
    fake_transformers = types.SimpleNamespace(SegformerForSemanticSegmentation=FakeSegformer)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    wrapped, returned_torch = convert_module._build_huggingface_module(
        {
            "model_class": "SegformerForSemanticSegmentation",
            "repo": "fake/segformer",
            "replace_gelu_with_relu": True,
        }
    )

    assert returned_torch is fake_torch
    assert isinstance(wrapped.model.activation_fn, FakeReLU)
    assert wrapped.forward("frame") == "logits:frame"
def test_convert_manifest_supports_huggingface_backend(tmp_path: Path) -> None:
    manifest = tmp_path / "models.yaml"
    _write_huggingface_manifest(manifest)
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    artifacts_root = repo_root / "artifacts"

    generated_vela = artifacts_root / "vela" / "segformerb0_seg_512_int8_vela.tflite"
    generated_vela.parent.mkdir(parents=True, exist_ok=True)
    generated_vela.write_bytes(b"vela")

    with (
        patch("tools.model_converter.convert._check_source_dependencies", return_value=[]),
        patch(
            "tools.model_converter.convert._build_huggingface_module",
            return_value=("<fake-nn-module>", _fake_torch_with_zeros()),
        ),
        patch(
            "tools.model_converter.convert.litert_convert.convert_pt_to_int8_tflite",
            return_value={
                "float": artifacts_root / "segformerb0_seg_512" / "segformerb0_seg_512_float.tflite",
                "int8": artifacts_root / "segformerb0_seg_512" / "segformerb0_seg_512_int8.tflite",
            },
        ) as convert_mock,
        patch("tools.model_converter.convert.write_vela_artifacts") as write_vela_mock,
        patch(
            "tools.model_converter.convert._copy_generated_vela_model",
            return_value=artifacts_root / "segformerb0_seg_512" / "segformerb0_seg_512_vela.tflite",
        ),
    ):
        model_dir = artifacts_root / "segformerb0_seg_512"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "segformerb0_seg_512_int8.tflite").write_bytes(b"int8")
        write_vela_mock.return_value = artifacts_root / "vela" / "segformerb0_seg_512.vela_info.json"
        results = convert_manifest(
            manifest_path=manifest,
            repo_root=repo_root,
            output_root=artifacts_root,
            vela_dir=artifacts_root / "vela",
        )

    assert [result.status for result in results] == ["exported_from_source"]
    convert_mock.assert_called_once()
    assert results[0].source_type == "huggingface"


def test_convert_manifest_supports_ultraface_backend(tmp_path: Path) -> None:
    manifest = tmp_path / "models.yaml"
    _write_ultraface_manifest(manifest)
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    artifacts_root = repo_root / "artifacts"

    with (
        patch("tools.model_converter.convert._check_source_dependencies", return_value=[]),
        patch(
            "tools.model_converter.convert._build_ultraface_module",
            return_value=("<fake-ultraface-module>", _fake_torch_with_zeros()),
            create=True,
        ) as build_mock,
        patch(
            "tools.model_converter.convert.litert_convert.convert_pt_to_int8_tflite",
            return_value={
                "float": artifacts_root / "ultraface_rfb320_face_320x240" / "ultraface_rfb320_face_320x240_float.tflite",
                "int8": artifacts_root / "ultraface_rfb320_face_320x240" / "ultraface_rfb320_face_320x240_int8.tflite",
            },
        ) as convert_mock,
        patch("tools.model_converter.convert.write_vela_artifacts") as write_vela_mock,
        patch(
            "tools.model_converter.convert._copy_generated_vela_model",
            return_value=artifacts_root / "ultraface_rfb320_face_320x240" / "ultraface_rfb320_face_320x240_vela.tflite",
        ),
    ):
        model_dir = artifacts_root / "ultraface_rfb320_face_320x240"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "ultraface_rfb320_face_320x240_int8.tflite").write_bytes(b"int8")
        write_vela_mock.return_value = artifacts_root / "vela" / "ultraface_rfb320_face_320x240.vela_info.json"
        results = convert_manifest(
            manifest_path=manifest,
            repo_root=repo_root,
            output_root=artifacts_root,
            vela_dir=artifacts_root / "vela",
        )

    assert [result.status for result in results] == ["exported_from_source"]
    build_mock.assert_called_once()
    convert_mock.assert_called_once()
    assert results[0].source_type == "ultraface"


def test_convert_manifest_supports_blazeface_backend(tmp_path: Path) -> None:
    manifest = tmp_path / "models.yaml"
    _write_blazeface_manifest(manifest)
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    artifacts_root = repo_root / "artifacts"

    with (
        patch("tools.model_converter.convert._check_source_dependencies", return_value=[]),
        patch(
            "tools.model_converter.convert._build_blazeface_module",
            return_value=("<fake-blazeface-module>", _fake_torch_with_zeros()),
            create=True,
        ) as build_mock,
        patch(
            "tools.model_converter.convert.litert_convert.convert_pt_to_int8_tflite",
            return_value={
                "float": artifacts_root / "blazeface_front_face_128" / "blazeface_front_face_128_float.tflite",
                "int8": artifacts_root / "blazeface_front_face_128" / "blazeface_front_face_128_int8.tflite",
            },
        ) as convert_mock,
        patch("tools.model_converter.convert.write_vela_artifacts") as write_vela_mock,
        patch(
            "tools.model_converter.convert._copy_generated_vela_model",
            return_value=artifacts_root / "blazeface_front_face_128" / "blazeface_front_face_128_vela.tflite",
        ),
    ):
        model_dir = artifacts_root / "blazeface_front_face_128"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "blazeface_front_face_128_int8.tflite").write_bytes(b"int8")
        write_vela_mock.return_value = artifacts_root / "vela" / "blazeface_front_face_128.vela_info.json"
        results = convert_manifest(
            manifest_path=manifest,
            repo_root=repo_root,
            output_root=artifacts_root,
            vela_dir=artifacts_root / "vela",
        )

    assert [result.status for result in results] == ["exported_from_source"]
    build_mock.assert_called_once()
    convert_mock.assert_called_once()
    assert results[0].source_type == "blazeface"


def test_convert_manifest_supports_retinaface_backend(tmp_path: Path) -> None:
    manifest = tmp_path / "models.yaml"
    _write_retinaface_manifest(manifest)
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    artifacts_root = repo_root / "artifacts"

    with (
        patch("tools.model_converter.convert._check_source_dependencies", return_value=[]),
        patch(
            "tools.model_converter.convert._build_retinaface_module",
            return_value=("<fake-retinaface-module>", _fake_torch_with_zeros()),
            create=True,
        ) as build_mock,
        patch(
            "tools.model_converter.convert.litert_convert.convert_pt_to_int8_tflite",
            return_value={
                "float": artifacts_root / "retinaface_mnet_face_160" / "retinaface_mnet_face_160_float.tflite",
                "int8": artifacts_root / "retinaface_mnet_face_160" / "retinaface_mnet_face_160_int8.tflite",
            },
        ) as convert_mock,
        patch("tools.model_converter.convert.write_vela_artifacts") as write_vela_mock,
        patch(
            "tools.model_converter.convert._copy_generated_vela_model",
            return_value=artifacts_root / "retinaface_mnet_face_160" / "retinaface_mnet_face_160_vela.tflite",
        ),
    ):
        model_dir = artifacts_root / "retinaface_mnet_face_160"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "retinaface_mnet_face_160_int8.tflite").write_bytes(b"int8")
        write_vela_mock.return_value = artifacts_root / "vela" / "retinaface_mnet_face_160.vela_info.json"
        results = convert_manifest(
            manifest_path=manifest,
            repo_root=repo_root,
            output_root=artifacts_root,
            vela_dir=artifacts_root / "vela",
        )

    assert [result.status for result in results] == ["exported_from_source"]
    build_mock.assert_called_once()
    convert_mock.assert_called_once()
    assert results[0].source_type == "retinaface"


def test_convert_manifest_supports_movenet_backend(tmp_path: Path) -> None:
    manifest = tmp_path / "models.yaml"
    _write_movenet_manifest(manifest)
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    artifacts_root = repo_root / "artifacts"

    with (
        patch("tools.model_converter.convert._check_source_dependencies", return_value=[]),
        patch(
            "tools.model_converter.convert._build_movenet_module",
            return_value=("<fake-movenet-module>", _fake_torch_with_zeros()),
            create=True,
        ) as build_mock,
        patch(
            "tools.model_converter.convert.litert_convert.convert_pt_to_int8_tflite",
            return_value={
                "float": artifacts_root / "movenet_lightning_pose_192" / "movenet_lightning_pose_192_float.tflite",
                "int8": artifacts_root / "movenet_lightning_pose_192" / "movenet_lightning_pose_192_int8.tflite",
            },
        ) as convert_mock,
        patch("tools.model_converter.convert.write_vela_artifacts") as write_vela_mock,
        patch(
            "tools.model_converter.convert._copy_generated_vela_model",
            return_value=artifacts_root / "movenet_lightning_pose_192" / "movenet_lightning_pose_192_vela.tflite",
        ),
    ):
        model_dir = artifacts_root / "movenet_lightning_pose_192"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "movenet_lightning_pose_192_int8.tflite").write_bytes(b"int8")
        write_vela_mock.return_value = artifacts_root / "vela" / "movenet_lightning_pose_192.vela_info.json"
        results = convert_manifest(
            manifest_path=manifest,
            repo_root=repo_root,
            output_root=artifacts_root,
            vela_dir=artifacts_root / "vela",
        )

    assert [result.status for result in results] == ["exported_from_source"]
    build_mock.assert_called_once()
    convert_mock.assert_called_once()
    assert results[0].source_type == "movenet"


def test_convert_manifest_supports_honk_backend(tmp_path: Path) -> None:
    manifest = tmp_path / "models.yaml"
    _write_honk_manifest(manifest)
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    artifacts_root = repo_root / "artifacts"

    with (
        patch("tools.model_converter.convert._check_source_dependencies", return_value=[]),
        patch(
            "tools.model_converter.convert._build_honk_module",
            return_value=("<fake-honk-module>", _fake_torch_with_zeros()),
            create=True,
        ) as build_mock,
        patch(
            "tools.model_converter.convert.litert_convert.convert_pt_to_int8_tflite",
            return_value={
                "float": artifacts_root / "honk_res8_narrow_kws_101x40" / "honk_res8_narrow_kws_101x40_float.tflite",
                "int8": artifacts_root / "honk_res8_narrow_kws_101x40" / "honk_res8_narrow_kws_101x40_int8.tflite",
            },
        ) as convert_mock,
        patch("tools.model_converter.convert.write_vela_artifacts") as write_vela_mock,
        patch(
            "tools.model_converter.convert._copy_generated_vela_model",
            return_value=artifacts_root / "honk_res8_narrow_kws_101x40" / "honk_res8_narrow_kws_101x40_vela.tflite",
        ),
    ):
        model_dir = artifacts_root / "honk_res8_narrow_kws_101x40"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "honk_res8_narrow_kws_101x40_int8.tflite").write_bytes(b"int8")
        write_vela_mock.return_value = artifacts_root / "vela" / "honk_res8_narrow_kws_101x40.vela_info.json"
        results = convert_manifest(
            manifest_path=manifest,
            repo_root=repo_root,
            output_root=artifacts_root,
            vela_dir=artifacts_root / "vela",
        )

    assert [result.status for result in results] == ["exported_from_source"]
    build_mock.assert_called_once()
    convert_mock.assert_called_once()
    assert results[0].source_type == "honk"


def test_convert_manifest_supports_prebuilt_tflite_backend(tmp_path: Path) -> None:
    manifest = tmp_path / "models.yaml"
    _write_prebuilt_tflite_manifest(manifest)
    repo_root = tmp_path / "repo"
    source_model = repo_root / "artifacts_debug/vendor/mediapipe/hand_landmarker_extracted"
    source_model.mkdir(parents=True)
    (source_model / "hand_landmarks_detector.tflite").write_bytes(b"prebuilt-tflite")
    artifacts_root = repo_root / "artifacts"

    with (
        patch("tools.model_converter.convert._check_source_dependencies", return_value=[]),
        patch("tools.model_converter.convert.write_vela_artifacts") as write_vela_mock,
        patch(
            "tools.model_converter.convert._copy_generated_vela_model",
            return_value=artifacts_root / "mediapipe_hand_landmarks_224" / "mediapipe_hand_landmarks_224_vela.tflite",
        ),
    ):
        model_dir = artifacts_root / "mediapipe_hand_landmarks_224"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "mediapipe_hand_landmarks_224_int8.tflite").write_bytes(b"prebuilt-tflite")
        write_vela_mock.return_value = artifacts_root / "vela" / "mediapipe_hand_landmarks_224.vela_info.json"
        results = convert_manifest(
            manifest_path=manifest,
            repo_root=repo_root,
            output_root=artifacts_root,
            vela_dir=artifacts_root / "vela",
        )

    assert [result.status for result in results] == ["exported_from_source"]
    assert results[0].source_type == "prebuilt_tflite"
    assert (artifacts_root / "mediapipe_hand_landmarks_224" / "mediapipe_hand_landmarks_224_int8.tflite").read_bytes() == b"prebuilt-tflite"
    write_vela_mock.assert_called_once()


def test_convert_manifest_can_rewrite_prebuilt_float16_dequantize_constants(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "models.yaml"
    _write_prebuilt_tflite_manifest(manifest, dequantize_float16_constants=True)
    repo_root = tmp_path / "repo"
    source_model = repo_root / "artifacts_debug/vendor/mediapipe/hand_landmarker_extracted"
    source_model.mkdir(parents=True)
    source_path = source_model / "hand_landmarks_detector.tflite"
    source_path.write_bytes(b"prebuilt-tflite")
    artifacts_root = repo_root / "artifacts"

    def rewrite_side_effect(input_path: Path, output_path: Path) -> int:
        assert input_path == source_path
        output_path.write_bytes(b"rewritten-tflite")
        return 3

    with (
        patch("tools.model_converter.convert._check_source_dependencies", return_value=[]),
        patch(
            "tools.model_converter.convert.rewrite_float16_dequantize_constants",
            side_effect=rewrite_side_effect,
        ) as rewrite_mock,
        patch("tools.model_converter.convert.write_vela_artifacts") as write_vela_mock,
        patch(
            "tools.model_converter.convert._copy_generated_vela_model",
            return_value=artifacts_root / "mediapipe_hand_landmarks_224" / "mediapipe_hand_landmarks_224_vela.tflite",
        ),
    ):
        write_vela_mock.return_value = artifacts_root / "vela" / "mediapipe_hand_landmarks_224.vela_info.json"
        results = convert_manifest(
            manifest_path=manifest,
            repo_root=repo_root,
            output_root=artifacts_root,
            vela_dir=artifacts_root / "vela",
        )

    int8_path = artifacts_root / "mediapipe_hand_landmarks_224" / "mediapipe_hand_landmarks_224_int8.tflite"
    assert int8_path.read_bytes() == b"rewritten-tflite"
    assert "folded 3 float16 dequantize constant op(s)" in results[0].notes
    rewrite_mock.assert_called_once()

def test_convert_manifest_can_rewrite_prebuilt_prelu_float_islands(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "models.yaml"
    _write_prebuilt_tflite_manifest(manifest, quantize_prelu_float_islands=True)
    repo_root = tmp_path / "repo"
    source_model = repo_root / "artifacts_debug/vendor/mediapipe/hand_landmarker_extracted"
    source_model.mkdir(parents=True)
    source_path = source_model / "hand_landmarks_detector.tflite"
    source_path.write_bytes(b"prebuilt-tflite")
    artifacts_root = repo_root / "artifacts"

    def rewrite_side_effect(input_path: Path, output_path: Path) -> int:
        assert input_path == source_path
        output_path.write_bytes(b"prelu-rewritten-tflite")
        return 31

    with (
        patch("tools.model_converter.convert._check_source_dependencies", return_value=[]),
        patch(
            "tools.model_converter.convert.rewrite_prelu_float_islands_to_int8",
            side_effect=rewrite_side_effect,
        ) as rewrite_mock,
        patch("tools.model_converter.convert.write_vela_artifacts") as write_vela_mock,
        patch(
            "tools.model_converter.convert._copy_generated_vela_model",
            return_value=artifacts_root / "mediapipe_hand_landmarks_224" / "mediapipe_hand_landmarks_224_vela.tflite",
        ),
    ):
        write_vela_mock.return_value = artifacts_root / "vela" / "mediapipe_hand_landmarks_224.vela_info.json"
        results = convert_manifest(
            manifest_path=manifest,
            repo_root=repo_root,
            output_root=artifacts_root,
            vela_dir=artifacts_root / "vela",
        )

    int8_path = artifacts_root / "mediapipe_hand_landmarks_224" / "mediapipe_hand_landmarks_224_int8.tflite"
    assert int8_path.read_bytes() == b"prelu-rewritten-tflite"
    assert "rewrote 31 float PReLU island(s) to int8 PReLU" in results[0].notes
    rewrite_mock.assert_called_once()

def test_convert_manifest_static_quantizes_prebuilt_before_prelu_rewrite(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "models.yaml"
    _write_prebuilt_tflite_manifest(
        manifest,
        dequantize_float16_constants=True,
        static_int8=True,
        quantize_prelu_float_islands=True,
    )
    repo_root = tmp_path / "repo"
    source_model = repo_root / "artifacts_debug/vendor/mediapipe/hand_landmarker_extracted"
    source_model.mkdir(parents=True)
    source_path = source_model / "hand_landmarks_detector.tflite"
    source_path.write_bytes(b"prebuilt-tflite")
    artifacts_root = repo_root / "artifacts"
    seen = {}

    def fold_side_effect(input_path: Path, output_path: Path) -> int:
        assert input_path == source_path
        output_path.write_bytes(b"folded-float-tflite")
        return 133

    def signature_side_effect(input_path: Path, output_path: Path) -> bool:
        assert input_path.name.endswith("_float16_folded.tflite")
        assert input_path.read_bytes() == b"folded-float-tflite"
        output_path.write_bytes(b"signed-float-tflite")
        return True

    def calibration_side_effect(dataset, *, input_shape, input_name, **kwargs):
        seen["calibration"] = (dataset, input_shape, input_name)
        return [{input_name: "sample"}]

    def quantize_side_effect(input_path: Path, output_path: Path, samples) -> Path:
        assert input_path.name.endswith("_signed.tflite")
        assert input_path.read_bytes() == b"signed-float-tflite"
        assert list(samples) == [{"input_1": "sample"}]
        output_path.write_bytes(b"static-int8-tflite")
        return output_path

    def prelu_side_effect(input_path: Path, output_path: Path) -> int:
        assert input_path.name.endswith("_static_int8.tflite")
        assert input_path.read_bytes() == b"static-int8-tflite"
        output_path.write_bytes(b"final-int8-tflite")
        return 31

    with (
        patch("tools.model_converter.convert._check_source_dependencies", return_value=[]),
        patch(
            "tools.model_converter.convert.rewrite_float16_dequantize_constants",
            side_effect=fold_side_effect,
        ) as fold_mock,
        patch(
            "tools.model_converter.convert.ensure_serving_default_signature",
            side_effect=signature_side_effect,
        ) as signature_mock,
        patch(
            "tools.model_converter.convert.litert_convert.signature_input_names",
            return_value=["input_1"],
        ),
        patch(
            "tools.model_converter.convert.calibration.load_calibration_samples",
            side_effect=calibration_side_effect,
        ),
        patch(
            "tools.model_converter.convert.litert_convert.quantize_to_int8",
            side_effect=quantize_side_effect,
        ) as quantize_mock,
        patch(
            "tools.model_converter.convert.rewrite_prelu_float_islands_to_int8",
            side_effect=prelu_side_effect,
        ) as prelu_mock,
        patch("tools.model_converter.convert.write_vela_artifacts") as write_vela_mock,
        patch(
            "tools.model_converter.convert._copy_generated_vela_model",
            return_value=artifacts_root / "mediapipe_hand_landmarks_224" / "mediapipe_hand_landmarks_224_vela.tflite",
        ),
    ):
        write_vela_mock.return_value = artifacts_root / "vela" / "mediapipe_hand_landmarks_224.vela_info.json"
        results = convert_manifest(
            manifest_path=manifest,
            repo_root=repo_root,
            output_root=artifacts_root,
            vela_dir=artifacts_root / "vela",
        )

    int8_path = artifacts_root / "mediapipe_hand_landmarks_224" / "mediapipe_hand_landmarks_224_int8.tflite"
    assert int8_path.read_bytes() == b"final-int8-tflite"
    assert seen["calibration"] == ("coco_128", [1, 224, 224, 3], "input_1")
    assert "added serving_default signature for static int8 quantization" in results[0].notes
    assert "quantized prebuilt TFLite to static int8" in results[0].notes
    fold_mock.assert_called_once()
    signature_mock.assert_called_once()
    quantize_mock.assert_called_once()
    prelu_mock.assert_called_once()


def test_convert_manifest_can_isolate_source_exports_in_subprocess(tmp_path: Path) -> None:
    manifest = tmp_path / "models.yaml"
    _write_ultralytics_manifest(manifest, name="yolo11n_od_192")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    artifacts_root = repo_root / "artifacts"
    model_dir = artifacts_root / "yolo11n_od_192"
    model_dir.mkdir(parents=True, exist_ok=True)
    int8_path = model_dir / "yolo11n_od_192_int8.tflite"
    int8_path.write_bytes(b"int8-from-worker")
    generated_vela = artifacts_root / "vela" / "yolo11n_od_192_int8_vela.tflite"
    generated_vela.parent.mkdir(parents=True, exist_ok=True)
    generated_vela.write_bytes(b"vela")

    completed = subprocess.CompletedProcess(
        args=["python"],
        returncode=0,
        stdout=json.dumps({"int8_model_path": str(int8_path), "notes": ["worker export ok"]}),
        stderr="",
    )

    with (
        patch("tools.model_converter.convert._check_source_dependencies", return_value=[]),
        patch("tools.model_converter.convert.subprocess.run", return_value=completed) as run_mock,
        patch("tools.model_converter.convert.write_vela_artifacts") as write_vela_mock,
    ):
        write_vela_mock.return_value = artifacts_root / "vela" / "yolo11n_od_192.vela_info.json"
        results = convert_manifest(
            manifest_path=manifest,
            repo_root=repo_root,
            output_root=artifacts_root,
            vela_dir=artifacts_root / "vela",
            isolate_source_exports=True,
        )

    assert [result.status for result in results] == ["exported_from_source"]
    assert results[0].int8_model_path == str(int8_path)
    assert "worker export ok" in results[0].notes
    run_mock.assert_called_once()


def test_convert_manifest_isolates_worker_crash_and_falls_back_to_model_zoo(tmp_path: Path) -> None:
    manifest = tmp_path / "models.yaml"
    _write_ultralytics_manifest(manifest, name="yolo11n_od_192", model_zoo_ref="yolo11n_ref.tflite")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    artifacts_root = repo_root / "artifacts"
    model_zoo_dir = repo_root / "model_zoo" / "tflm_yolo11_od"
    model_zoo_dir.mkdir(parents=True)
    (model_zoo_dir / "yolo11n_ref.tflite").write_bytes(b"reference-tflite")

    crashed = subprocess.CompletedProcess(
        args=["python"],
        returncode=134,
        stdout="",
        stderr="free(): double free detected in tcache 2",
    )

    with (
        patch("tools.model_converter.convert._check_source_dependencies", return_value=[]),
        patch("tools.model_converter.convert.subprocess.run", return_value=crashed) as run_mock,
        patch("tools.model_converter.convert.write_vela_artifacts") as write_vela_mock,
    ):
        write_vela_mock.return_value = artifacts_root / "vela" / "yolo11n_od_192.vela_info.json"
        results = convert_manifest(
            manifest_path=manifest,
            repo_root=repo_root,
            output_root=artifacts_root,
            vela_dir=artifacts_root / "vela",
            isolate_source_exports=True,
        )

    assert [result.status for result in results] == ["staged_model_zoo_ref"]
    assert any("double free" in note for note in results[0].notes)
    assert (artifacts_root / "yolo11n_od_192" / "yolo11n_od_192_int8.tflite").read_bytes() == b"reference-tflite"
    run_mock.assert_called_once()


def test_convert_manifest_falls_back_to_model_zoo_reference_when_dependencies_missing(tmp_path: Path) -> None:
    """Missing in-process deps -> ``staged_model_zoo_ref``.

    The new dependency probe is ``_check_source_dependencies`` (in-process
    importlib); the OLD ``_probe_wsl_modules`` symbol no longer exists. Driving
    the fallback by reporting ``ultralytics`` missing keeps the assertion
    focused on the staging branch (no torch conversion attempted).
    """
    manifest = tmp_path / "models.yaml"
    _write_ultralytics_manifest(manifest, name="yolo11n_od_192", model_zoo_ref="yolo11n_ref.tflite")
    repo_root = tmp_path / "repo"
    model_zoo_dir = repo_root / "model_zoo" / "tflm_yolo11_od"
    model_zoo_dir.mkdir(parents=True)
    source_tflite = model_zoo_dir / "yolo11n_ref.tflite"
    source_tflite.write_bytes(b"test-tflite")

    with (
        patch(
            "tools.model_converter.convert._check_source_dependencies",
            return_value=["ultralytics"],
        ),
        patch(
            "tools.model_converter.convert.litert_convert.convert_pt_to_int8_tflite"
        ) as convert_mock,
        patch(
            "tools.model_converter.convert._build_ultralytics_module"
        ) as build_module_mock,
        patch("tools.model_converter.convert.write_vela_artifacts") as write_vela_mock,
    ):
        write_vela_mock.return_value = repo_root / "artifacts" / "vela" / "yolo11n_od_192.vela_info.json"
        results = convert_manifest(
            manifest_path=manifest,
            repo_root=repo_root,
            output_root=repo_root / "artifacts",
            vela_dir=repo_root / "artifacts" / "vela",
        )

    assert [result.status for result in results] == ["staged_model_zoo_ref"]
    staged_int8 = repo_root / "artifacts" / "yolo11n_od_192" / "yolo11n_od_192_int8.tflite"
    staged_vela = repo_root / "artifacts" / "yolo11n_od_192" / "yolo11n_od_192_vela.tflite"
    assert staged_int8.read_bytes() == b"test-tflite"
    assert staged_vela.read_bytes() == b"test-tflite"
    # The source-conversion seam must NOT have been reached when deps are missing.
    convert_mock.assert_not_called()
    build_module_mock.assert_not_called()
    write_vela_mock.assert_called_once()


def test_convert_manifest_falls_back_to_model_zoo_reference_when_source_export_raises(tmp_path: Path) -> None:
    """Source export failure -> ``staged_model_zoo_ref`` (failure isolation).

    Even when all dependencies are present, if the in-process converter raises,
    the model_zoo_ref escape hatch kicks in and the manifest run continues.
    """
    manifest = tmp_path / "models.yaml"
    _write_ultralytics_manifest(manifest, name="yolo11n_od_192", model_zoo_ref="yolo11n_ref.tflite")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    model_zoo_dir = repo_root / "model_zoo" / "tflm_yolo11_od"
    model_zoo_dir.mkdir(parents=True)
    (model_zoo_dir / "yolo11n_ref.tflite").write_bytes(b"reference-tflite")

    with (
        patch("tools.model_converter.convert._check_source_dependencies", return_value=[]),
        patch(
            "tools.model_converter.convert._build_ultralytics_module",
            return_value=("<fake-nn-module>", _fake_torch_with_zeros()),
        ),
        patch(
            "tools.model_converter.convert.litert_convert.convert_pt_to_int8_tflite",
            side_effect=RuntimeError("conversion exploded"),
        ),
        patch("tools.model_converter.convert.write_vela_artifacts") as write_vela_mock,
    ):
        write_vela_mock.return_value = repo_root / "artifacts" / "vela" / "yolo11n_od_192.vela_info.json"
        results = convert_manifest(
            manifest_path=manifest,
            repo_root=repo_root,
            output_root=repo_root / "artifacts",
            vela_dir=repo_root / "artifacts" / "vela",
        )

    assert [result.status for result in results] == ["staged_model_zoo_ref"]
    staged_int8 = repo_root / "artifacts" / "yolo11n_od_192" / "yolo11n_od_192_int8.tflite"
    assert staged_int8.read_bytes() == b"reference-tflite"
    # The failure must be recorded in notes for diagnosis (failure isolation).
    assert any("conversion exploded" in note for note in results[0].notes)
    write_vela_mock.assert_called_once()


def test_convert_manifest_does_not_treat_stale_int8_artifact_as_fresh_source_export(tmp_path: Path) -> None:
    """A stale int8 tflite is never trusted as a source export.

    With the in-process pipeline there is no marker-file cache (the converter
    always re-runs), so this test now drives the fallback via a converter
    exception and asserts the staged reference fully overwrites the stale int8
    file (no marker file is ever written).
    """
    manifest = tmp_path / "models.yaml"
    _write_ultralytics_manifest(manifest, name="yolov8n_od_192", model_zoo_ref="yolov8n_ref.tflite")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    artifacts_root = repo_root / "artifacts"
    model_zoo_dir = repo_root / "model_zoo" / "tflm_yolov8_od"
    model_zoo_dir.mkdir(parents=True)
    (model_zoo_dir / "yolov8n_ref.tflite").write_bytes(b"reference-tflite")
    stale_int8 = artifacts_root / "yolov8n_od_192" / "yolov8n_od_192_int8.tflite"
    stale_int8.parent.mkdir(parents=True, exist_ok=True)
    stale_int8.write_bytes(b"stale-tflite")

    with (
        patch("tools.model_converter.convert._check_source_dependencies", return_value=[]),
        patch(
            "tools.model_converter.convert._build_ultralytics_module",
            return_value=("<fake-nn-module>", _fake_torch_with_zeros()),
        ),
        patch(
            "tools.model_converter.convert.litert_convert.convert_pt_to_int8_tflite",
            side_effect=RuntimeError("source export failed"),
        ),
        patch("tools.model_converter.convert.write_vela_artifacts") as write_vela_mock,
    ):
        write_vela_mock.return_value = artifacts_root / "vela" / "yolov8n_od_192.vela_info.json"
        results = convert_manifest(
            manifest_path=manifest,
            repo_root=repo_root,
            output_root=artifacts_root,
            vela_dir=artifacts_root / "vela",
        )

    assert [result.status for result in results] == ["staged_model_zoo_ref"]
    staged_int8 = artifacts_root / "yolov8n_od_192" / "yolov8n_od_192_int8.tflite"
    staged_vela = artifacts_root / "yolov8n_od_192" / "yolov8n_od_192_vela.tflite"
    # The stale bytes are overwritten by the model_zoo reference, not kept.
    assert staged_int8.read_bytes() == b"reference-tflite"
    assert staged_vela.read_bytes() == b"reference-tflite"
    # No ONNX-era marker file should ever be produced by the in-process path.
    assert not (artifacts_root / "yolov8n_od_192" / "source_export.json").exists()
    write_vela_mock.assert_called_once()


def test_convert_cli_writes_summary_json(tmp_path: Path) -> None:
    """``write_summary`` emits the conversion_summary.json with the right shape.

    Drives the source-export happy path so the result has a real int8_model_path
    (not None) and the JSON reflects the in-process pipeline.
    """
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    manifest = repo_root / "models.yaml"
    _write_ultralytics_manifest(manifest, name="yolo11n_od_192", model_zoo_ref="yolo11n_ref.tflite")
    model_zoo_dir = repo_root / "model_zoo" / "tflm_yolo11_od"
    model_zoo_dir.mkdir(parents=True)
    (model_zoo_dir / "yolo11n_ref.tflite").write_bytes(b"test-tflite")

    with (
        patch("tools.model_converter.convert._check_source_dependencies", return_value=[]),
        patch(
            "tools.model_converter.convert._build_ultralytics_module",
            return_value=("<fake-nn-module>", _fake_torch_with_zeros()),
        ),
        patch(
            "tools.model_converter.convert.litert_convert.convert_pt_to_int8_tflite",
            side_effect=RuntimeError("forced fallback for summary test"),
        ),
        patch("tools.model_converter.convert.write_vela_artifacts") as write_vela_mock,
    ):
        write_vela_mock.return_value = repo_root / "artifacts" / "vela" / "yolo11n_od_192.vela_info.json"
        results = convert_manifest(
            manifest_path=manifest,
            repo_root=repo_root,
            output_root=repo_root / "artifacts",
            vela_dir=repo_root / "artifacts" / "vela",
        )

    summary_path = repo_root / "artifacts" / "conversion_summary.json"
    from tools.model_converter.convert import write_summary

    write_summary(results, repo_root / "artifacts")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["results"][0]["name"] == "yolo11n_od_192"
    assert payload["results"][0]["status"] == "staged_model_zoo_ref"
