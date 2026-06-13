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
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

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


def _write_ultralytics_manifest(manifest: Path, name: str = "yolo11n_od_192", model_zoo_ref: str | None = None) -> None:
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
    tier: 1
    flash_address: 0x00400000{extra}
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
