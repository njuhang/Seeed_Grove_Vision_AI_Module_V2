"""Model-conversion orchestrator.

Dispatches Phase-1 models through the **in-process litert-torch** conversion
pipeline (Task 1.1 + 1.2). The orchestrator runs in the WSL base env, where
``torch`` / ``litert_torch`` / ``ai_edge_quantizer`` and ``ultralytics`` are all
importable, so conversion happens **in-process** -- there is no ``wsl``
subprocess, no ONNX, and no ``onnx2bf``/``onnx2tf`` step.

Per-model flow
--------------
1. Resolve the source backend from ``spec.source["type"]``.
   * ``ultralytics``: build a pre-NMS ``nn.Module`` from ``YOLO(weights).model``
     and convert it via ``litert_convert.convert_pt_to_int8_tflite`` with
     representative samples from ``calibration.load_calibration_samples``.
   * other source types (torchvision / timm / torch.hub / HF): Task 1.4 -- they
     raise ``NotImplementedError`` here and are reported as
     ``unsupported_source_type``.
2. Optionally compile the int8 tflite with vela (``--skip-vela`` bypasses it).
3. If conversion fails AND ``allow_model_zoo_ref`` is set AND a
   ``model_zoo_ref`` exists, stage the reference tflite as a fallback
   (``staged_model_zoo_ref``). This is the model_zoo escape hatch.

Each model is wrapped in its own try/except so one model's failure never blocks
the rest of the manifest.
"""

import argparse
import importlib
import json
import shutil
import sys
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.model_converter import calibration
from tools.model_converter import litert_convert
from tools.model_converter.model_registry import ModelSpec, load_models
from tools.model_converter.quantize import artifact_dir
from tools.model_converter.quantize import quantized_model_path
from tools.model_converter.quantize import vela_model_path
from tools.model_converter.vela_compile import write_vela_artifacts

# Per-source-type modules the in-process backend needs to import before the
# conversion can run. Used purely for a dependency probe -- the actual imports
# happen lazily inside the backend so the orchestrator (and the hermetic test
# suite) can import this module without torch/ultralytics installed.
REQUIRED_SOURCE_DEPS: dict[str, tuple[str, ...]] = {
    "ultralytics": ("torch", "litert_torch", "ai_edge_quantizer", "ultralytics"),
    "torchvision": ("torch", "litert_torch", "ai_edge_quantizer", "torchvision"),
    "timm": ("torch", "litert_torch", "ai_edge_quantizer", "timm"),
    "torch.hub": ("torch", "litert_torch", "ai_edge_quantizer"),
    "huggingface": ("torch", "litert_torch", "ai_edge_quantizer", "transformers"),
}

# Source backends that are wired up in this phase. Anything else is reported as
# ``unsupported_source_type`` (Task 1.4 implements the rest).
IMPLEMENTED_SOURCE_TYPES: frozenset[str] = frozenset({"ultralytics"})


@dataclass(frozen=True)
class ConversionResult:
    name: str
    task: str
    status: str
    source_type: str
    input_shape: list[int]
    output_dir: str
    int8_model_path: str | None
    vela_model_path: str | None
    vela_info_path: str | None
    source_artifact_path: str | None
    model_zoo_ref: str | None
    missing_dependencies: list[str]
    notes: list[str]


def iter_phase1_models(manifest_path: str = "configs/models.yaml") -> list[str]:
    return [model.name for model in load_models(manifest_path) if model.tier == 1]


def _select_models(models: list[ModelSpec], names: set[str] | None) -> list[ModelSpec]:
    phase1 = [model for model in models if model.tier == 1]
    if names is None:
        return phase1

    selected = [model for model in phase1 if model.name in names]
    missing = sorted(names - {model.name for model in selected})
    if missing:
        raise ValueError(f"unknown model(s): {', '.join(missing)}")
    return selected


def _resolve_model_zoo_reference(repo_root: Path, spec: ModelSpec) -> Path | None:
    if not spec.model_zoo_ref:
        return None

    matches = list((repo_root / "model_zoo").rglob(spec.model_zoo_ref))
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one model_zoo match for {spec.name}: {spec.model_zoo_ref}, got {len(matches)}"
        )
    return matches[0]


def _check_source_dependencies(modules: tuple[str, ...]) -> list[str]:
    """Return the subset of ``modules`` not importable in this process.

    Replaces the old ``_probe_wsl_modules`` (which shelled out to ``wsl``). The
    conversion now runs in-process in the orchestrator's env, so the probe is a
    plain ``importlib.util.find_spec`` against the current interpreter.
    """
    missing: list[str] = []
    for name in modules:
        try:
            if importlib.util.find_spec(name) is None:
                missing.append(name)
        except (ImportError, ModuleNotFoundError):
            missing.append(name)
        except ValueError:
            # find_spec raises ValueError for relative names; treat as missing.
            missing.append(name)
    return missing


def _stage_reference_artifacts(
    spec: ModelSpec,
    repo_root: Path,
    output_root: Path,
) -> tuple[Path, Path, Path]:
    source_path = _resolve_model_zoo_reference(repo_root, spec)
    if source_path is None:
        raise ValueError(f"model_zoo_ref is required to stage existing artifact for {spec.name}")

    model_output_dir = artifact_dir(spec.name, output_root)
    model_output_dir.mkdir(parents=True, exist_ok=True)

    int8_path = quantized_model_path(spec.name, output_root)
    vela_path = vela_model_path(spec.name, output_root)
    shutil.copy2(source_path, int8_path)
    shutil.copy2(source_path, vela_path)
    return source_path, int8_path, vela_path


def _build_ultralytics_module(weights: str) -> tuple[Any, Any]:
    """Construct a pre-NMS ``nn.Module`` from an ultralytics checkpoint.

    ``YOLO(weights).model`` is the full ``DetectionModel`` -- an ``nn.Module``
    whose forward returns a ``(raw_predictions, [post-processing dict])`` tuple
    in eval mode. litert-torch requires a single-tensor-output module to trace,
    so we wrap it in a tiny ``nn.Module`` that returns ``out[0]`` -- the raw
    pre-NMS head tensor. This matches the ``nopost`` semantics used by the
    model_zoo reference models (post-processing happens on-device in firmware).

    Returns ``(module, sample_input)`` where ``sample_input`` is a zero tensor
    shaped to the spec's ``input_shape``. The caller is responsible for the
    actual ``input_shape`` (we return a placeholder here and rebuild the sample
    inside the dispatcher where the spec is in scope).
    """
    import torch  # noqa: PLC0415 -- lazy: orchestrator imports must stay light.
    from ultralytics import YOLO  # noqa: PLC0415

    detection_model = YOLO(weights).model
    detection_model.eval()

    class _PreNMSWrapper(torch.nn.Module):
        """Return only the raw pre-NMS head output from a DetectionModel."""

        def __init__(self, inner: torch.nn.Module) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, x: Any) -> Any:
            out = self.inner(x)
            if isinstance(out, (list, tuple)):
                return out[0]
            return out

    module = _PreNMSWrapper(detection_model).eval()
    return module, torch


def _convert_from_source(
    spec: ModelSpec,
    repo_root: Path,
    output_root: Path,
    vela_dir: Path,
    skip_vela: bool,
) -> tuple[Path, Path | None, Path | None, list[str]]:
    """Run the in-process litert-torch conversion for a single model.

    Dispatches on ``spec.source["type"]``. Only ``ultralytics`` is wired up in
    this phase; anything else raises ``NotImplementedError`` (Task 1.4).
    """
    notes: list[str] = []
    source_type = spec.source["type"]

    if source_type != "ultralytics":
        raise NotImplementedError(
            f"source type '{source_type}' does not have an implemented "
            "conversion backend yet (Task 1.4)"
        )

    # Build the pre-NMS nn.Module from the ultralytics checkpoint.
    module, torch = _build_ultralytics_module(spec.source["model"])
    sample_args = (torch.zeros(*spec.input_shape),)

    # Calibration samples keyed by the signature input arg name. litert-torch
    # emits ``args_0`` for the single-positional trace (verified against real
    # yolo11n -- see test_calibration.test_real_quantize_to_int8_accepts_loader_output
    # and the in-env smoke). Using the loader's default keeps this in sync with
    # Task 1.2's contract; quantize_to_int8 would surface a key-mismatch if the
    # signature ever diverged.
    calibration_samples = calibration.load_calibration_samples(
        spec.calibration_dataset,
        input_shape=spec.input_shape,
    )

    paths = litert_convert.convert_pt_to_int8_tflite(
        module,
        sample_args,
        spec.name,
        output_root=output_root,
        calibration_samples=calibration_samples,
    )
    int8_path = paths["int8"]
    notes.append("exported source model through in-process litert-torch int8 pipeline")

    vela_path: Path | None = None
    vela_info_path: Path | None = None
    if not skip_vela:
        vela_info_path = write_vela_artifacts(
            model_name=spec.name,
            model_path=int8_path,
            output_dir=vela_dir,
        )
        vela_path = _copy_generated_vela_model(spec.name, int8_path, output_root, vela_dir)
        notes.append("generated vela-optimized tflite and sidecar metadata")

    return int8_path, vela_path, vela_info_path, notes


def _copy_generated_vela_model(
    model_name: str,
    int8_path: Path,
    output_root: Path,
    vela_dir: Path,
) -> Path:
    generated = sorted(vela_dir.glob(f"{int8_path.stem}*_vela.tflite"))
    if len(generated) != 1:
        raise ValueError(
            f"expected exactly one vela model for {model_name} from {int8_path.stem}, got {len(generated)}"
        )
    final_path = vela_model_path(model_name, output_root)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(generated[0], final_path)
    return final_path


def convert_model(
    spec: ModelSpec,
    repo_root: str | Path = ".",
    output_root: str | Path = "artifacts",
    vela_dir: str | Path = "artifacts/vela",
    skip_vela: bool = False,
    allow_model_zoo_ref: bool = True,
) -> ConversionResult:
    repo_root = Path(repo_root).resolve()
    output_root = Path(output_root).resolve()
    vela_dir = Path(vela_dir).resolve()
    model_output_dir = artifact_dir(spec.name, output_root)
    model_output_dir.mkdir(parents=True, exist_ok=True)

    notes: list[str] = []
    source_artifact_path: str | None = None
    int8_path: Path | None = None
    vela_path: Path | None = None
    vela_info_path: Path | None = None

    source_type = spec.source["type"]
    required = REQUIRED_SOURCE_DEPS.get(source_type, ())
    missing_dependencies = _check_source_dependencies(required)

    source_supported = source_type in IMPLEMENTED_SOURCE_TYPES
    deps_ready = source_supported and not missing_dependencies

    if deps_ready:
        try:
            int8_path, vela_path, vela_info_path, source_notes = _convert_from_source(
                spec=spec,
                repo_root=repo_root,
                output_root=output_root,
                vela_dir=vela_dir,
                skip_vela=skip_vela,
            )
            notes.extend(source_notes)
            return ConversionResult(
                name=spec.name,
                task=spec.task,
                status="exported_from_source",
                source_type=source_type,
                input_shape=spec.input_shape,
                output_dir=str(model_output_dir),
                int8_model_path=str(int8_path),
                vela_model_path=str(vela_path) if vela_path else None,
                vela_info_path=str(vela_info_path) if vela_info_path else None,
                source_artifact_path=None,
                model_zoo_ref=spec.model_zoo_ref,
                missing_dependencies=[],
                notes=notes,
            )
        except Exception as exc:
            notes.append(f"source export failed: {exc}")
            if allow_model_zoo_ref and spec.model_zoo_ref:
                source_path, int8_path, vela_path = _stage_reference_artifacts(spec, repo_root, output_root)
                source_artifact_path = str(source_path)
                notes.append("fell back to existing model_zoo reference")
                if not skip_vela:
                    vela_info_path = write_vela_artifacts(
                        model_name=spec.name,
                        model_path=vela_path,
                        output_dir=vela_dir,
                    )
                    notes.append("generated vela sidecar from staged tflite")
                return ConversionResult(
                    name=spec.name,
                    task=spec.task,
                    status="staged_model_zoo_ref",
                    source_type=source_type,
                    input_shape=spec.input_shape,
                    output_dir=str(model_output_dir),
                    int8_model_path=str(int8_path),
                    vela_model_path=str(vela_path),
                    vela_info_path=str(vela_info_path) if vela_info_path else None,
                    source_artifact_path=source_artifact_path,
                    model_zoo_ref=spec.model_zoo_ref,
                    missing_dependencies=[],
                    notes=notes,
                )
            status = "export_failed"
    elif not source_supported:
        notes.append(f"source type '{source_type}' does not have an implemented conversion backend yet")
        status = "unsupported_source_type"
    else:
        notes.append("conversion environment is missing required packages for source export")
        status = "blocked_missing_dependencies"

    if allow_model_zoo_ref and spec.model_zoo_ref and status in {"blocked_missing_dependencies", "unsupported_source_type"}:
        source_path, int8_path, vela_path = _stage_reference_artifacts(spec, repo_root, output_root)
        source_artifact_path = str(source_path)
        notes.append("fell back to existing model_zoo reference")
        if not skip_vela:
            vela_info_path = write_vela_artifacts(
                model_name=spec.name,
                model_path=vela_path,
                output_dir=vela_dir,
            )
            notes.append("generated vela sidecar from staged tflite")
        return ConversionResult(
            name=spec.name,
            task=spec.task,
            status="staged_model_zoo_ref",
            source_type=source_type,
            input_shape=spec.input_shape,
            output_dir=str(model_output_dir),
            int8_model_path=str(int8_path),
            vela_model_path=str(vela_path),
            vela_info_path=str(vela_info_path) if vela_info_path else None,
            source_artifact_path=source_artifact_path,
            model_zoo_ref=spec.model_zoo_ref,
            missing_dependencies=missing_dependencies,
            notes=notes,
        )

    return ConversionResult(
        name=spec.name,
        task=spec.task,
        status=status,
        source_type=source_type,
        input_shape=spec.input_shape,
        output_dir=str(model_output_dir),
        int8_model_path=str(int8_path) if int8_path else None,
        vela_model_path=str(vela_path) if vela_path else None,
        vela_info_path=str(vela_info_path) if vela_info_path else None,
        source_artifact_path=source_artifact_path,
        model_zoo_ref=spec.model_zoo_ref,
        missing_dependencies=missing_dependencies,
        notes=notes,
    )


def convert_manifest(
    manifest_path: str | Path = "configs/models.yaml",
    repo_root: str | Path = ".",
    output_root: str | Path = "artifacts",
    vela_dir: str | Path = "artifacts/vela",
    model_names: list[str] | None = None,
    skip_vela: bool = False,
    allow_model_zoo_ref: bool = True,
) -> list[ConversionResult]:
    manifest_path = Path(manifest_path)
    models = load_models(manifest_path)
    selected = _select_models(models, set(model_names) if model_names else None)
    return [
        convert_model(
            spec=model,
            repo_root=repo_root,
            output_root=output_root,
            vela_dir=vela_dir,
            skip_vela=skip_vela,
            allow_model_zoo_ref=allow_model_zoo_ref,
        )
        for model in selected
    ]


def write_summary(results: list[ConversionResult], output_root: str | Path = "artifacts") -> Path:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "conversion_summary.json"
    payload = {"results": [asdict(result) for result in results]}
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="configs/models.yaml")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--vela-dir", default="artifacts/vela")
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--skip-vela", action="store_true")
    parser.add_argument("--no-model-zoo-ref", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = convert_manifest(
        manifest_path=args.manifest,
        repo_root=args.repo_root,
        output_root=args.output_dir,
        vela_dir=args.vela_dir,
        model_names=args.models,
        skip_vela=args.skip_vela,
        allow_model_zoo_ref=not args.no_model_zoo_ref,
    )
    summary_path = write_summary(results, args.output_dir)

    for result in results:
        print(json.dumps(asdict(result), ensure_ascii=True))
    print(summary_path)

    if any(result.status.startswith("staged_") for result in results):
        return 0
    if all(
        result.status
        in {"blocked_missing_dependencies", "unsupported_source_type", "export_failed"}
        for result in results
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
