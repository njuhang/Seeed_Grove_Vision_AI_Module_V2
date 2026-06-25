from pathlib import Path

import pytest

from tools.model_converter.model_registry import load_models


def test_load_models_manifest(tmp_path: Path) -> None:
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
    flash_address: 0x400000
        """.strip(),
        encoding="utf-8",
    )

    models = load_models(manifest)

    assert [model.name for model in models] == ["yolo11n_od_192"]
    assert models[0].flash_address == 0x400000


def test_rejects_unaligned_flash_address(tmp_path: Path) -> None:
    manifest = tmp_path / "bad.yaml"
    manifest.write_text(
        """
models:
  - name: broken_model
    task: object_detection
    source:
      type: ultralytics
      model: broken.pt
    input_shape: [1, 3, 192, 192]
    calibration_dataset: coco_128
    tier: 1
    flash_address: 0x400123
        """.strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="4KB aligned"):
        load_models(manifest)


def test_rejects_duplicate_model_names(tmp_path: Path) -> None:
    manifest = tmp_path / "dup.yaml"
    manifest.write_text(
        """
models:
  - name: duplicate_model
    task: object_detection
    source:
      type: ultralytics
      model: a.pt
    input_shape: [1, 3, 192, 192]
    calibration_dataset: coco_128
    tier: 1
    flash_address: 0x400000
  - name: duplicate_model
    task: classification
    source:
      type: torchvision
      model: mobilenet_v2
    input_shape: [1, 3, 224, 224]
    calibration_dataset: imagenet_1k_random
    tier: 1
    flash_address: 0x600000
        """.strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate model name"):
        load_models(manifest)


def test_rejects_overlapping_flash_slots(tmp_path: Path) -> None:
    manifest = tmp_path / "overlap.yaml"
    manifest.write_text(
        """
models:
  - name: first_model
    task: object_detection
    source:
      type: ultralytics
      model: a.pt
    input_shape: [1, 3, 192, 192]
    calibration_dataset: coco_128
    tier: 1
    flash_address: 0x400000
  - name: second_model
    task: classification
    source:
      type: torchvision
      model: mobilenet_v2
    input_shape: [1, 3, 224, 224]
    calibration_dataset: imagenet_1k_random
    tier: 1
    flash_address: 0x500000
        """.strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="flash slots overlap"):
        load_models(manifest)


def test_repo_manifest_has_multiple_tier1_models() -> None:
    manifest = Path("configs/models.yaml")

    models = load_models(manifest)
    names = {model.name for model in models if model.tier == 1}

    assert {
        "yolo11n_od_192",
        "yolov8n_od_192",
        "yolov5n_od_192",
        "mobilenetv2_cls_224",
        "mobilenetv3small_cls_224",
        "efficientnet_lite0_cls_224",
        "efficientnet_lite1_cls_240",
        "deeplabv3_mbnv3_seg_320",
        "segformerb0_seg_512",
        "yolov8n_pose_192",
        "yolo11n_pose_192",
        "movenet_lightning_pose_192",
        "ultraface_rfb320_face_320x240",
        "blazeface_front_face_128",
        "retinaface_mnet_face_160",
        "honk_res8_narrow_kws_101x40",
        "mediapipe_hand_detector_192",
        "mediapipe_hand_landmarks_224",
        "lraspp_mbnv3_seg_192",
    }.issubset(names)

    lraspp_fit = next(model for model in models if model.name == "lraspp_mbnv3_seg_192")
    assert lraspp_fit.input_shape == [1, 3, 192, 192]
    assert lraspp_fit.vela is not None
    assert lraspp_fit.vela.extra_args == ("--arena-cache-size", "0")

    lraspp_320 = next(model for model in models if model.name == "lraspp_mbnv3_seg_320")
    assert lraspp_320.vela is not None
    assert lraspp_320.vela.extra_args == ("--arena-cache-size", "0")

def test_repo_manifest_configures_lraspp_segmentation_fit_variants() -> None:
    models = load_models(Path("configs/models.yaml"))

    lraspp_320 = next(model for model in models if model.name == "lraspp_mbnv3_seg_320")
    assert lraspp_320.vela is not None
    assert lraspp_320.vela.extra_args == ("--arena-cache-size", "0")

    lraspp_fit = next(model for model in models if model.name == "lraspp_mbnv3_seg_192")
    assert lraspp_fit.input_shape == [1, 3, 192, 192]
    assert lraspp_fit.vela is not None
    assert lraspp_fit.vela.extra_args == ("--arena-cache-size", "0")


def test_repo_manifest_configures_phase6_yolo11_size_variants() -> None:
    models = load_models(Path("configs/models.yaml"))
    by_name = {model.name: model for model in models}

    assert by_name["yolo11s_od_192"].tier == 2
    assert by_name["yolo11s_od_192"].source["model"] == "yolo11s.pt"
    assert "yolo11m_od_192" not in by_name


def test_repo_manifest_configures_phase6_yolo_fastest_candidate() -> None:
    models = load_models(Path("configs/models.yaml"))
    by_name = {model.name: model for model in models}

    candidate = by_name["yolo_fastestv2_od_192"]
    assert candidate.tier == 2
    assert candidate.task == "object_detection"
    assert candidate.input_shape == [1, 3, 192, 192]
    assert candidate.source["type"] == "yolo_fastestv2"
    assert candidate.source["repo"] == "dog-qiuqiu/Yolo-FastestV2"


def test_repo_manifest_configures_2stage_face_recognition_pipeline() -> None:
    models = load_models(Path("configs/models.yaml"))
    by_name = {model.name: model for model in models}

    detector = by_name["scrfd_500m_face_det_240x320"]
    embedder = by_name["mobilefacenet_face_embed_112"]

    assert detector.tier == 3
    assert detector.task == "face_detection"
    assert detector.source["type"] == "prebuilt_tflite"
    assert detector.source["pipeline_id"] == "face_recognition_2stage_scrfd_mobilefacenet"
    assert detector.source["pipeline_stage"] == 1
    assert detector.input_shape == [1, 3, 240, 320]

    assert embedder.tier == 3
    assert embedder.task == "face_embedding"
    assert embedder.source["type"] == "prebuilt_tflite"
    assert embedder.source["pipeline_id"] == detector.source["pipeline_id"]
    assert embedder.source["pipeline_stage"] == 2
    assert embedder.input_shape == [1, 3, 112, 112]

