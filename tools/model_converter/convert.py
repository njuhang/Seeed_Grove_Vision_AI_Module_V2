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
from importlib import util as importlib_util
import json
import subprocess
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
from tools.model_converter.tflite_rewrite import ensure_serving_default_signature
from tools.model_converter.tflite_rewrite import rewrite_float16_dequantize_constants
from tools.model_converter.tflite_rewrite import rewrite_int8_conv_biases_to_int32
from tools.model_converter.tflite_rewrite import rewrite_prelu_float_islands_to_int8
from tools.model_converter.vela_compile import write_vela_artifacts

# Per-source-type modules the in-process backend needs to import before the
# conversion can run. Used purely for a dependency probe -- the actual imports
# happen lazily inside the backend so the orchestrator (and the hermetic test
# suite) can import this module without torch/ultralytics installed.
REQUIRED_SOURCE_DEPS: dict[str, tuple[str, ...]] = {
    "ultralytics": ("torch", "litert_torch", "ai_edge_quantizer", "ultralytics"),
    "torchvision": ("torch", "litert_torch", "ai_edge_quantizer", "torchvision"),
    "timm": ("torch", "litert_torch", "ai_edge_quantizer", "timm"),
    "torch.hub": ("torch", "litert_torch", "ai_edge_quantizer", "pandas", "seaborn"),
    "huggingface": ("torch", "litert_torch", "ai_edge_quantizer", "transformers"),
    "ultraface": ("torch", "litert_torch", "ai_edge_quantizer"),
    "blazeface": ("torch", "litert_torch", "ai_edge_quantizer"),
    "retinaface": ("torch", "litert_torch", "ai_edge_quantizer", "torchvision"),
    "movenet": ("torch", "litert_torch", "ai_edge_quantizer"),
    "honk": ("torch", "litert_torch", "ai_edge_quantizer"),
    "yolo_fastestv2": ("torch", "litert_torch", "ai_edge_quantizer"),
    "prebuilt_tflite": (),
}

# Source backends that are wired up in this phase. Anything else is reported as
# ``unsupported_source_type`` (Task 1.4 implements the rest).
IMPLEMENTED_SOURCE_TYPES: frozenset[str] = frozenset({"ultralytics", "torchvision", "timm", "torch.hub", "huggingface", "ultraface", "blazeface", "retinaface", "movenet", "honk", "yolo_fastestv2", "prebuilt_tflite"})


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


def parse_tiers(raw: str) -> set[int]:
    tiers = {int(item.strip()) for item in raw.split(",") if item.strip()}
    if not tiers:
        raise ValueError("--tiers must include at least one integer tier")
    return tiers


def iter_phase1_models(manifest_path: str = "configs/models.yaml") -> list[str]:
    return [model.name for model in load_models(manifest_path) if model.tier == 1]


def _select_models(
    models: list[ModelSpec],
    names: set[str] | None,
    *,
    tiers: set[int] | None = None,
) -> list[ModelSpec]:
    selected_tiers = tiers or {1}
    tier_models = [model for model in models if model.tier in selected_tiers]
    if names is None:
        return tier_models

    selected = [model for model in tier_models if model.name in names]
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
            if importlib_util.find_spec(name) is None:
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

    module = _wrap_ultralytics_raw_head_module(detection_model, torch)
    return module, torch


class _UltralyticsRawHeadWrapper:
    """Return a stable, export-friendly detect-head output from ultralytics.

    For modern Ultralytics detect heads (YOLOv8 / YOLO11), `head(x)` in eval
    mode returns a tuple `(decoded_preds, aux_dict)` where `decoded_preds` is
    the no-NMS tensor we want for benchmark parity with the repo's model_zoo
    references, and `aux_dict` contains large intermediate feature maps that
    should *not* escape into the exported TFLite outputs.

    For YOLOv5, the eval path decodes boxes via `(wh * 2) ** 2`, which lowers
    to `tfl.pow` and fails LiteRT legalization. That path is still special-
    cased below by forcing the training branch and returning the raw per-scale
    detect outputs instead.
    """

    def __init__(self, inner: Any) -> None:
        self.inner = inner

    def eval(self) -> "_UltralyticsRawHeadWrapper":
        if hasattr(self.inner, "eval"):
            self.inner.eval()
        return self

    def forward(self, x: Any) -> Any:
        y: list[Any] = []
        for module in self.inner.model:
            if module.f != -1:
                x = y[module.f] if isinstance(module.f, int) else [x if j == -1 else y[j] for j in module.f]

            is_last_head = module is self.inner.model[-1] and hasattr(module, "forward_head")
            is_yolov5_detect_head = (
                module is self.inner.model[-1]
                and not hasattr(module, "forward_head")
                and hasattr(module, "training")
                and hasattr(module, "export")
                and hasattr(module, "nl")
            )
            if is_last_head:
                decoded = module(x)
                if isinstance(decoded, (list, tuple)) and decoded:
                    x = decoded[0]
                else:
                    x = decoded
            elif is_yolov5_detect_head:
                # YOLOv5's Detect head decodes boxes in eval mode via
                # `(wh * 2) ** 2`, which lowers to `tfl.pow` and fails LiteRT
                # legalization. Its training branch returns the raw per-scale
                # head outputs before that decode step, which matches the
                # "nopost" semantics we want for board-side post-processing.
                previous_training = module.training
                try:
                    module.training = True
                    x = module(x)
                finally:
                    module.training = previous_training
            else:
                x = module(x)

            y.append(x if module.i in self.inner.save else None)
        return x


def _wrap_ultralytics_raw_head_module(inner: Any, torch: Any) -> Any:
    helper = _UltralyticsRawHeadWrapper(inner)

    class _TorchModuleWrapper(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.helper = helper

        def forward(self, x: Any) -> Any:
            return self.helper.forward(x)

    return _TorchModuleWrapper().eval()


def _build_torchvision_module(source: dict[str, Any]) -> tuple[Any, Any]:
    """Construct a torchvision model from the manifest source spec.

    The manifest stores ``source.model`` as the factory name under
    ``torchvision.models`` and may optionally carry a ``weights`` enum path such
    as ``MobileNet_V2_Weights.DEFAULT``. torchvision classification models
    usually return a single tensor directly, while segmentation models return an
    ``OrderedDict`` like ``{"out": ..., "aux": ...}``. Wrap them behind a
    single-tensor facade so litert-torch always traces one tensor output.
    """
    import torch  # noqa: PLC0415 -- lazy import for hermetic tests.
    import torchvision.models as tv_models  # noqa: PLC0415

    model_name = source["model"]
    namespaces: list[Any] = [tv_models]
    for attr in ("segmentation", "detection"):
        namespace = getattr(tv_models, attr, None)
        if namespace is not None:
            namespaces.append(namespace)

    factory = None
    for namespace in namespaces:
        candidate = getattr(namespace, model_name, None)
        if candidate is not None:
            factory = candidate
            break
    if factory is None:
        raise AttributeError(f"torchvision model {model_name!r} not found in known namespaces")

    weights = None
    weights_spec = source.get("weights")
    if weights_spec:
        enum_name, member_name = weights_spec.split(".", 1)
        enum_owner = None
        for namespace in namespaces:
            candidate = getattr(namespace, enum_name, None)
            if candidate is not None:
                enum_owner = candidate
                break
        if enum_owner is None:
            raise AttributeError(f"torchvision weights enum {enum_name!r} not found in known namespaces")
        weights = getattr(enum_owner, member_name)

    inner = factory(weights=weights).eval()

    if _is_torchvision_detection_module(inner):
        return _wrap_torchvision_detection_raw_head_module(inner, torch), torch

    class _SingleTensorOutputWrapper(torch.nn.Module):
        """Normalize torchvision outputs to a single tensor.

        Segmentation heads commonly return ``{"out": tensor, "aux": tensor}``
        while classification heads return a bare tensor. Prefer ``out`` when it
        is present and otherwise leave single-tensor outputs untouched.
        """

        def __init__(self, model: torch.nn.Module) -> None:
            super().__init__()
            self.model = model

        def forward(self, x: Any) -> Any:
            out = self.model(x)
            if isinstance(out, dict):
                if "out" in out:
                    return out["out"]
                if "logits" in out:
                    return out["logits"]
                if len(out) == 1:
                    return next(iter(out.values()))
            if isinstance(out, (list, tuple)) and len(out) == 1:
                return out[0]
            return out

    return _SingleTensorOutputWrapper(inner).eval(), torch


def _is_torchvision_detection_module(module: Any) -> bool:
    return all(
        hasattr(module, attr)
        for attr in ("transform", "backbone", "head", "postprocess_detections")
    )


def _wrap_torchvision_detection_raw_head_module(inner: Any, torch: Any) -> Any:
    """Bypass torchvision detection post-processing for export.

    SSD/SSDLite ``forward`` enters anchor decoding and NMS, which introduces
    data-dependent shapes that ``torch.export`` cannot guard. The benchmark only
    needs raw model execution, so expose the pre-NMS head tensor instead.
    """

    class _DetectionRawHeadWrapper(torch.nn.Module):
        def __init__(self, model: torch.nn.Module) -> None:
            super().__init__()
            self.model = model

        def forward(self, x: Any) -> Any:
            images, _ = self.model.transform(x, None)
            features = self.model.backbone(images.tensors)
            if hasattr(features, "values"):
                features = list(features.values())
            elif not isinstance(features, (list, tuple)):
                features = [features]

            head_outputs = self.model.head(features)
            return head_outputs["bbox_regression"], head_outputs["cls_logits"]

    return _DetectionRawHeadWrapper(inner).eval()


def _build_timm_module(source: dict[str, Any]) -> tuple[Any, Any]:
    """Construct a timm model from the manifest source spec."""
    import torch  # noqa: PLC0415
    import timm  # noqa: PLC0415

    module = timm.create_model(
        source["model"],
        pretrained=bool(source.get("pretrained", True)),
    ).eval()
    return module, torch


def _build_torch_hub_module(source: dict[str, Any]) -> tuple[Any, Any]:
    """Construct a torch.hub model from the manifest source spec.

    ``torch.hub`` may prompt to trust a GitHub repo the first time it is used.
    Batch conversion runs non-interactively inside a worker subprocess, so we
    default to ``trust_repo=True`` / ``skip_validation=True`` unless the
    manifest overrides them explicitly.
    """
    import torch  # noqa: PLC0415

    extra_kwargs = dict(source.get("kwargs", {}))
    module = torch.hub.load(
        source["repo"],
        source["model"],
        pretrained=bool(source.get("pretrained", True)),
        trust_repo=source.get("trust_repo", True),
        autoshape=source.get("autoshape", False),
        skip_validation=bool(source.get("skip_validation", True)),
        **extra_kwargs,
    )
    if hasattr(torch, "nn") and hasattr(module, "model") and hasattr(module.model, "model"):
        return _wrap_ultralytics_raw_head_module(module.model, torch), torch
    return module.eval(), torch


def _replace_modules_by_class_name(module: Any, replacements: dict[str, Any]) -> None:
    """Recursively replace child modules whose class names match replacements."""
    named_children = getattr(module, "named_children", None)
    if named_children is None:
        return
    for name, child in list(named_children()):
        replacement_factory = replacements.get(child.__class__.__name__)
        if replacement_factory is not None:
            setattr(module, name, replacement_factory())
        else:
            _replace_modules_by_class_name(child, replacements)


def _build_huggingface_module(source: dict[str, Any]) -> tuple[Any, Any]:
    """Construct a HuggingFace vision model and expose a single-tensor output.

    Transformers vision heads often return dataclass-like outputs instead of a
    bare tensor. litert-torch tracing is happier with a plain tensor, so we wrap
    the model and return ``.logits`` when available.
    """
    import torch  # noqa: PLC0415
    import transformers  # noqa: PLC0415

    model_class = getattr(transformers, source["model_class"])
    inner = model_class.from_pretrained(source["repo"]).eval()
    if source.get("replace_gelu_with_relu"):
        _replace_modules_by_class_name(inner, {"GELUActivation": torch.nn.ReLU})

    class _LogitsWrapper(torch.nn.Module):
        def __init__(self, model: torch.nn.Module) -> None:
            super().__init__()
            self.model = model

        def forward(self, x: Any) -> Any:
            out = self.model(x)
            if hasattr(out, "logits"):
                return out.logits
            if isinstance(out, dict) and "logits" in out:
                return out["logits"]
            return out

    return _LogitsWrapper(inner).eval(), torch


def _resolve_repo_relative_path(repo_root: Path, value: str | Path, *, field_name: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"{field_name} does not exist: {path}")
    return path


def _build_ultraface_module(source: dict[str, Any], repo_root: Path) -> tuple[Any, Any]:
    """Construct UltraFace from the upstream PyTorch implementation.

    UltraFace's test-mode forward decodes priors and applies Softmax. The
    benchmark firmware measures raw model execution, so load the network with
    ``is_test=False`` and export the raw ``(confidences, locations)`` heads.
    """
    import torch  # noqa: PLC0415

    upstream_root = _resolve_repo_relative_path(
        repo_root,
        source["repo_path"],
        field_name="source.repo_path",
    )
    weights_path = _resolve_repo_relative_path(
        upstream_root,
        source["weights"],
        field_name="source.weights",
    )

    sys.path.insert(0, str(upstream_root))
    try:
        from vision.ssd.config.fd_config import define_img_size  # noqa: PLC0415
        from vision.ssd.mb_tiny_fd import create_mb_tiny_fd  # noqa: PLC0415
        from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd  # noqa: PLC0415

        define_img_size(int(source.get("input_size", 320)))
        labels_path = upstream_root / source.get("label_file", "models/voc-model-labels.txt")
        if labels_path.exists():
            num_classes = len([line for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()])
        else:
            num_classes = int(source.get("num_classes", 2))

        variant = str(source.get("variant", "RFB")).lower()
        if variant == "rfb":
            module = create_Mb_Tiny_RFB_fd(num_classes, is_test=False, device="cpu")
        elif variant == "slim":
            module = create_mb_tiny_fd(num_classes, is_test=False, device="cpu")
        else:
            raise ValueError(f"unsupported UltraFace variant: {source.get('variant')!r}")
        module.load(str(weights_path))
        return module.eval(), torch
    finally:
        try:
            sys.path.remove(str(upstream_root))
        except ValueError:
            pass


def _build_blazeface_module(source: dict[str, Any], repo_root: Path) -> tuple[Any, Any]:
    """Construct BlazeFace from the community PyTorch conversion.

    ``BlazeFace.forward`` returns raw regression and confidence tensors. The
    decode/NMS path lives under ``predict_on_batch`` and is intentionally not
    exported for this benchmark.
    """
    import torch  # noqa: PLC0415

    upstream_root = _resolve_repo_relative_path(
        repo_root,
        source["repo_path"],
        field_name="source.repo_path",
    )
    weights_path = _resolve_repo_relative_path(
        upstream_root,
        source["weights"],
        field_name="source.weights",
    )

    sys.path.insert(0, str(upstream_root))
    try:
        from blazeface import BlazeFace  # noqa: PLC0415

        model_name = str(source.get("model", "front")).lower()
        if model_name == "front":
            module = BlazeFace(back_model=False)
        elif model_name == "back":
            module = BlazeFace(back_model=True)
        else:
            raise ValueError(f"unsupported BlazeFace model: {source.get('model')!r}")
        module.load_weights(str(weights_path))
        return module.eval(), torch
    finally:
        try:
            sys.path.remove(str(upstream_root))
        except ValueError:
            pass


def _build_retinaface_module(source: dict[str, Any], repo_root: Path) -> tuple[Any, Any]:
    """Construct RetinaFace-MobileNet and expose raw detection heads."""
    import copy  # noqa: PLC0415
    import torch  # noqa: PLC0415

    upstream_root = _resolve_repo_relative_path(
        repo_root,
        source["repo_path"],
        field_name="source.repo_path",
    )
    weights_path = _resolve_repo_relative_path(
        upstream_root,
        source["weights"],
        field_name="source.weights",
    )

    sys.path.insert(0, str(upstream_root))
    try:
        from data.config import cfg_mnet  # noqa: PLC0415
        from models.retinaface import RetinaFace  # noqa: PLC0415

        backbone = str(source.get("backbone", "mobile0.25")).lower()
        if backbone not in {"mobile0.25", "mobilenet0.25", "mnet"}:
            raise ValueError(f"unsupported RetinaFace backbone: {source.get('backbone')!r}")

        cfg = copy.deepcopy(cfg_mnet)
        cfg["pretrain"] = False
        module = RetinaFace(cfg=cfg, phase="train")

        state_dict = torch.load(weights_path, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if not isinstance(state_dict, dict):
            raise ValueError(f"unsupported RetinaFace checkpoint format: {weights_path}")
        state_dict = {str(key).removeprefix("module."): value for key, value in state_dict.items()}
        missing, unexpected = module.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise ValueError(
                f"RetinaFace checkpoint mismatch: missing={len(missing)}, unexpected={len(unexpected)}"
            )
        return module.eval(), torch
    finally:
        try:
            sys.path.remove(str(upstream_root))
        except ValueError:
            pass


def _build_movenet_module(source: dict[str, Any], repo_root: Path) -> tuple[Any, Any]:
    """Construct MoveNet from the standalone PyTorch model file.

    The upstream repo's package-level ``lib`` import pulls in training data
    dependencies. Load the model file directly so conversion only needs the
    inference graph and checkpoint.
    """
    import torch  # noqa: PLC0415

    upstream_root = _resolve_repo_relative_path(
        repo_root,
        source["repo_path"],
        field_name="source.repo_path",
    )
    weights_path = _resolve_repo_relative_path(
        upstream_root,
        source["weights"],
        field_name="source.weights",
    )
    model_file = _resolve_repo_relative_path(
        upstream_root,
        source.get("model_file", "lib/models/movenet_mobilenetv2.py"),
        field_name="source.model_file",
    )

    module_spec = importlib_util.spec_from_file_location("movenet_model_file", model_file)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"could not load MoveNet model file: {model_file}")
    movenet_module = importlib_util.module_from_spec(module_spec)
    module_spec.loader.exec_module(movenet_module)

    model_class = getattr(movenet_module, source.get("class_name", "MoveNet"))
    module = model_class(
        num_classes=int(source.get("num_classes", 17)),
        width_mult=float(source.get("width_mult", 1.0)),
        mode=source.get("mode", "train"),
    )

    state_dict = torch.load(weights_path, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if not isinstance(state_dict, dict):
        raise ValueError(f"unsupported MoveNet checkpoint format: {weights_path}")
    state_dict = {str(key).removeprefix("module."): value for key, value in state_dict.items()}
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"MoveNet checkpoint mismatch: missing={len(missing)}, unexpected={len(unexpected)}"
        )

    class _TupleOutputWrapper(torch.nn.Module):
        def __init__(self, model: torch.nn.Module) -> None:
            super().__init__()
            self.model = model

        def forward(self, x: Any) -> Any:
            out = self.model(x)
            if isinstance(out, list):
                return tuple(out)
            return out

    return _TupleOutputWrapper(module).eval(), torch


def _build_honk_module(source: dict[str, Any], repo_root: Path) -> tuple[Any, Any]:
    """Construct Honk's Speech Commands Res8-narrow KWS model."""
    import torch  # noqa: PLC0415

    architecture = str(source.get("architecture", "res8-narrow"))
    if architecture != "res8-narrow":
        raise ValueError(f"unsupported Honk architecture: {architecture!r}")

    upstream_root = _resolve_repo_relative_path(
        repo_root,
        source["repo_path"],
        field_name="source.repo_path",
    )
    weights_path = _resolve_repo_relative_path(
        upstream_root,
        source["weights"],
        field_name="source.weights",
    )

    class _HonkRes8Narrow(torch.nn.Module):
        def __init__(self, n_labels: int) -> None:
            super().__init__()
            n_maps = 19
            self.conv0 = torch.nn.Conv2d(1, n_maps, (3, 3), padding=(1, 1), bias=False)
            self.pool = torch.nn.AvgPool2d((4, 3))
            for index in range(1, 7):
                setattr(
                    self,
                    f"conv{index}",
                    torch.nn.Conv2d(n_maps, n_maps, (3, 3), padding=1, dilation=1, bias=False),
                )
                setattr(self, f"bn{index}", torch.nn.BatchNorm2d(n_maps, affine=False))
            self.final_pool = torch.nn.AvgPool2d((25, 13))
            self.output = torch.nn.Linear(n_maps, n_labels)

        def forward(self, x: Any) -> Any:
            x = x.unsqueeze(1)
            old_x = None
            for index in range(7):
                y = torch.relu(getattr(self, f"conv{index}")(x))
                if index == 0:
                    y = self.pool(y)
                    old_x = y
                elif index % 2 == 0:
                    y = y + old_x
                    old_x = y
                if index > 0:
                    y = getattr(self, f"bn{index}")(y)
                x = y
            x = self.final_pool(x)
            x = torch.flatten(x, 1)
            return self.output(x)

    module = _HonkRes8Narrow(n_labels=int(source.get("n_labels", 12)))
    state_dict = torch.load(weights_path, map_location="cpu")
    if not isinstance(state_dict, dict):
        raise ValueError(f"unsupported Honk checkpoint format: {weights_path}")
    state_dict = {str(key).removeprefix("module."): value for key, value in state_dict.items()}
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"Honk checkpoint mismatch: missing={len(missing)}, unexpected={len(unexpected)}"
        )
    return module.eval(), torch


def _load_simple_datafile(path: Path) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    int_keys = {"classes", "width", "height", "anchor_num"}
    list_keys = {"anchors", "steps"}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("[") or "=" not in stripped:
            continue
        key, value = (part.strip() for part in stripped.split("=", 1))
        if key in int_keys:
            cfg[key] = int(value)
        elif key in list_keys:
            cfg[key] = [float(item) for item in value.split(",") if item.strip()]
        else:
            cfg[key] = value
    return cfg


def _build_yolo_fastestv2_module(source: dict[str, Any], repo_root: Path) -> tuple[Any, Any]:
    """Construct YOLO-FastestV2 and expose its raw two-head output.

    The upstream project includes post-processing utilities, but Phase 6 only
    benchmarks model execution. Export the raw detector heads and leave decode
    / NMS outside the TFLite graph.
    """
    import types  # noqa: PLC0415
    import torch  # noqa: PLC0415

    upstream_root = _resolve_repo_relative_path(
        repo_root,
        source["repo_path"],
        field_name="source.repo_path",
    )
    weights_path = _resolve_repo_relative_path(
        upstream_root,
        source["weights"],
        field_name="source.weights",
    )
    data_path = _resolve_repo_relative_path(
        upstream_root,
        source.get("data", "data/coco.data"),
        field_name="source.data",
    )
    cfg = _load_simple_datafile(data_path)

    sys.modules.setdefault("torchsummary", types.SimpleNamespace(summary=lambda *args, **kwargs: None))
    sys.path.insert(0, str(upstream_root))
    try:
        from model.detector import Detector  # noqa: PLC0415

        module = Detector(int(cfg.get("classes", 80)), int(cfg.get("anchor_num", 3)), True)
        state_dict = torch.load(weights_path, map_location="cpu")
        if not isinstance(state_dict, dict):
            raise ValueError(f"unsupported YOLO-FastestV2 checkpoint format: {weights_path}")
        state_dict = {str(key).removeprefix("module."): value for key, value in state_dict.items()}
        missing, unexpected = module.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise ValueError(
                "YOLO-FastestV2 checkpoint mismatch: "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )
        return module.eval(), torch
    finally:
        try:
            sys.path.remove(str(upstream_root))
        except ValueError:
            pass


def _channel_last_shape(input_shape: list[int]) -> list[int]:
    if len(input_shape) != 4:
        raise ValueError(
            f"channel-last export expects a 4D image tensor, got shape {input_shape!r}"
        )
    batch, channels, height, width = (int(dim) for dim in input_shape)
    return [batch, height, width, channels]


def _should_use_channel_last_io(spec: ModelSpec) -> bool:
    return spec.task == "classification" and len(spec.input_shape) == 4


def _rewrite_global_adaptive_avg_pool2d_as_mean(module: Any, torch: Any) -> Any:
    """Replace ``AdaptiveAvgPool2d(1)`` with an equivalent spatial mean op.

    litert-torch currently lowers some ``AdaptiveAvgPool2d(1)`` sites to
    ``GATHER_ND`` in the exported TFLite graph, which then survive Vela as CPU
    fallback ops. For fixed-shape classification exports we only need the exact
    ``output_size == 1`` behavior, which is semantically equivalent to a mean
    over the spatial axes and has a simpler lowering.
    """

    adaptive_avg_pool2d = getattr(torch.nn, "AdaptiveAvgPool2d", None)
    if adaptive_avg_pool2d is None or not hasattr(module, "named_children"):
        return module

    class _SpatialMeanPool2d(torch.nn.Module):
        def forward(self, x: Any) -> Any:
            return x.mean(dim=(-2, -1), keepdim=True)

    def _is_global_pool(child: Any) -> bool:
        if adaptive_avg_pool2d is None or not isinstance(child, adaptive_avg_pool2d):
            return False
        output_size = getattr(child, "output_size", None)
        return output_size == 1 or tuple(output_size) == (1, 1)

    for name, child in module.named_children():
        replacement = child
        if _is_global_pool(child):
            replacement = _SpatialMeanPool2d()
        else:
            replacement = _rewrite_global_adaptive_avg_pool2d_as_mean(child, torch)
        if replacement is not child:
            setattr(module, name, replacement)

    return module


def _convert_from_source(
    spec: ModelSpec,
    repo_root: Path,
    output_root: Path,
    vela_dir: Path,
    skip_vela: bool,
) -> tuple[Path, Path | None, Path | None, list[str]]:
    """Run the in-process litert-torch conversion for a single model.

    Dispatches on ``spec.source["type"]``. ``ultralytics`` and ``torchvision``
    are wired up in this phase; anything else raises ``NotImplementedError``
    (Task 1.4).
    """
    int8_path, notes = _export_int8_from_source(
        spec=spec,
        repo_root=repo_root,
        output_root=output_root,
    )

    vela_path: Path | None = None
    vela_info_path: Path | None = None
    if not skip_vela:
        vela_info_path = write_vela_artifacts(
            model_name=spec.name,
            model_path=int8_path,
            output_dir=vela_dir,
            **_build_vela_kwargs(spec),
        )
        vela_path = _copy_generated_vela_model(spec.name, int8_path, output_root, vela_dir)
        notes.append("generated vela-optimized tflite and sidecar metadata")

    return int8_path, vela_path, vela_info_path, notes


def _export_int8_from_source(
    spec: ModelSpec,
    repo_root: Path,
    output_root: Path,
) -> tuple[Path, list[str]]:
    notes: list[str] = []
    source_type = spec.source["type"]

    if source_type == "prebuilt_tflite":
        source_path = _resolve_repo_relative_path(
            repo_root,
            spec.source["path"],
            field_name="source.path",
        )
        model_output_dir = artifact_dir(spec.name, output_root)
        model_output_dir.mkdir(parents=True, exist_ok=True)
        int8_path = quantized_model_path(spec.name, output_root)
        current_path = source_path
        if spec.source.get("dequantize_float16_constants"):
            fold_output_path = int8_path
            if spec.source.get("quantize_prelu_float_islands"):
                fold_output_path = model_output_dir / f"{spec.name}_float16_folded.tflite"
            rewritten_count = rewrite_float16_dequantize_constants(
                current_path, fold_output_path
            )
            notes.append(
                f"folded {rewritten_count} float16 dequantize constant op(s)"
            )
            current_path = fold_output_path

        if spec.source.get("static_int8"):
            signed_path = model_output_dir / f"{spec.name}_signed.tflite"
            added_signature = ensure_serving_default_signature(current_path, signed_path)
            if added_signature:
                notes.append("added serving_default signature for static int8 quantization")
            current_path = signed_path

            input_names = litert_convert.signature_input_names(current_path)
            input_name = input_names[0] if input_names else calibration.DEFAULT_INPUT_NAME
            calibration_samples = calibration.load_calibration_samples(
                spec.calibration_dataset,
                input_shape=_prebuilt_tflite_calibration_shape(spec),
                input_name=input_name,
            )
            static_int8_path = int8_path
            if spec.source.get("quantize_prelu_float_islands"):
                static_int8_path = model_output_dir / f"{spec.name}_static_int8.tflite"
            litert_convert.quantize_to_int8(
                current_path,
                static_int8_path,
                calibration_samples,
            )
            notes.append("quantized prebuilt TFLite to static int8")
            current_path = static_int8_path

        if spec.source.get("rewrite_int8_conv_biases"):
            bias_rewrite_path = int8_path
            if spec.source.get("quantize_prelu_float_islands"):
                bias_rewrite_path = model_output_dir / f"{spec.name}_int32_bias.tflite"
            rewritten_count = rewrite_int8_conv_biases_to_int32(
                current_path, bias_rewrite_path
            )
            notes.append(
                f"rewrote {rewritten_count} int8 Conv/DepthwiseConv bias tensor(s) to int32"
            )
            current_path = bias_rewrite_path

        if spec.source.get("quantize_prelu_float_islands"):
            rewritten_count = rewrite_prelu_float_islands_to_int8(
                current_path, int8_path
            )
            notes.append(
                f"rewrote {rewritten_count} float PReLU island(s) to int8 PReLU"
            )
        elif current_path != int8_path:
            shutil.copy2(current_path, int8_path)
            notes.append("staged official prebuilt TFLite through the standard artifact path")
        return int8_path, notes

    # CPU-only converter bootstrap must happen before any backend imports torch
    # / torchvision / ultralytics so TensorFlow and torch_xla2 inherit the
    # intended process-wide environment.
    litert_convert._bootstrap_litert_env()

    if source_type == "ultralytics":
        module, torch = _build_ultralytics_module(spec.source["model"])
    elif source_type == "torchvision":
        module, torch = _build_torchvision_module(spec.source)
    elif source_type == "timm":
        module, torch = _build_timm_module(spec.source)
    elif source_type == "torch.hub":
        module, torch = _build_torch_hub_module(spec.source)
    elif source_type == "huggingface":
        module, torch = _build_huggingface_module(spec.source)
    elif source_type == "ultraface":
        module, torch = _build_ultraface_module(spec.source, repo_root)
    elif source_type == "blazeface":
        module, torch = _build_blazeface_module(spec.source, repo_root)
    elif source_type == "retinaface":
        module, torch = _build_retinaface_module(spec.source, repo_root)
    elif source_type == "movenet":
        module, torch = _build_movenet_module(spec.source, repo_root)
    elif source_type == "honk":
        module, torch = _build_honk_module(spec.source, repo_root)
    elif source_type == "yolo_fastestv2":
        module, torch = _build_yolo_fastestv2_module(spec.source, repo_root)
    else:
        raise NotImplementedError(
            f"source type '{source_type}' does not have an implemented "
            "conversion backend yet (Task 1.4)"
        )
    export_module = module
    export_input_shape = list(spec.input_shape)
    if _should_use_channel_last_io(spec):
        export_module = _rewrite_global_adaptive_avg_pool2d_as_mean(export_module, torch)
        litert_torch = importlib.import_module("litert_torch")
        export_module = litert_torch.to_channel_last_io(export_module, args=[0])
        export_input_shape = _channel_last_shape(spec.input_shape)
        notes.append("rewrote AdaptiveAvgPool2d(1) to spatial mean for export stability")
        notes.append("wrapped classification model for channel-last IO export")
    if hasattr(export_module, "eval"):
        export_module = export_module.eval()

    sample_args = (torch.zeros(*export_input_shape),)

    calibration_samples = calibration.load_calibration_samples(
        spec.calibration_dataset,
        input_shape=export_input_shape,
    )

    paths = litert_convert.convert_pt_to_int8_tflite(
        export_module,
        sample_args,
        spec.name,
        output_root=output_root,
        calibration_samples=calibration_samples,
    )
    int8_path = paths["int8"]
    notes.append("exported source model through in-process litert-torch int8 pipeline")
    return int8_path, notes



def _prebuilt_tflite_calibration_shape(spec: ModelSpec) -> list[int]:
    shape = [int(dim) for dim in spec.input_shape]
    if spec.source.get("input_layout") == "NHWC" and len(shape) == 4:
        return [shape[0], shape[2], shape[3], shape[1]]
    return shape
def _export_int8_from_source_subprocess(
    spec: ModelSpec,
    manifest_path: Path,
    repo_root: Path,
    output_root: Path,
) -> tuple[Path, list[str]]:
    command = [
        sys.executable,
        "-m",
        "tools.model_converter.source_export_worker",
        "--manifest",
        str(manifest_path),
        "--model",
        spec.name,
        "--repo-root",
        str(repo_root),
        "--output-root",
        str(output_root),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or f"worker exited with {result.returncode}"
        raise RuntimeError(f"source export worker failed with exit code {result.returncode}: {message}")

    stdout_lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not stdout_lines:
        raise RuntimeError("source export worker produced no JSON result")
    payload = json.loads(stdout_lines[-1])
    return Path(payload["int8_model_path"]), list(payload.get("notes", []))


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


def _build_vela_kwargs(spec: ModelSpec) -> dict[str, Any]:
    if spec.vela is None:
        return {}

    vela_kwargs: dict[str, Any] = {}
    if spec.vela.executable is not None:
        vela_kwargs["vela_executable"] = spec.vela.executable
    if spec.vela.executable_args:
        vela_kwargs["vela_executable_args"] = list(spec.vela.executable_args)
    if spec.vela.accelerator_config is not None:
        vela_kwargs["accelerator_config"] = spec.vela.accelerator_config
    if spec.vela.config is not None:
        vela_kwargs["config_path"] = spec.vela.config
    if spec.vela.system_config is not None:
        vela_kwargs["system_config"] = spec.vela.system_config
    if spec.vela.memory_mode is not None:
        vela_kwargs["memory_mode"] = spec.vela.memory_mode
    if spec.vela.optimise is not None:
        vela_kwargs["optimise"] = spec.vela.optimise
    if spec.vela.tensor_allocator is not None:
        vela_kwargs["tensor_allocator"] = spec.vela.tensor_allocator
    if spec.vela.extra_args:
        vela_kwargs["extra_args"] = list(spec.vela.extra_args)
    return vela_kwargs


def convert_model(
    spec: ModelSpec,
    manifest_path: str | Path = "configs/models.yaml",
    repo_root: str | Path = ".",
    output_root: str | Path = "artifacts",
    vela_dir: str | Path = "artifacts/vela",
    skip_vela: bool = False,
    allow_model_zoo_ref: bool = True,
    isolate_source_exports: bool = False,
) -> ConversionResult:
    manifest_path = Path(manifest_path).resolve()
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
            if isolate_source_exports:
                int8_path, source_notes = _export_int8_from_source_subprocess(
                    spec=spec,
                    manifest_path=manifest_path,
                    repo_root=repo_root,
                    output_root=output_root,
                )
                vela_path = None
                vela_info_path = None
                if not skip_vela:
                    vela_info_path = write_vela_artifacts(
                        model_name=spec.name,
                        model_path=int8_path,
                        output_dir=vela_dir,
                        **_build_vela_kwargs(spec),
                    )
                    vela_path = _copy_generated_vela_model(spec.name, int8_path, output_root, vela_dir)
                    source_notes = [
                        *source_notes,
                        "generated vela-optimized tflite and sidecar metadata",
                    ]
            else:
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
                        **_build_vela_kwargs(spec),
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
                **_build_vela_kwargs(spec),
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
    isolate_source_exports: bool = False,
    tiers: set[int] | None = None,
) -> list[ConversionResult]:
    manifest_path = Path(manifest_path)
    models = load_models(manifest_path)
    selected = _select_models(models, set(model_names) if model_names else None, tiers=tiers)
    return [
        convert_model(
            spec=model,
            manifest_path=manifest_path,
            repo_root=repo_root,
            output_root=output_root,
            vela_dir=vela_dir,
            skip_vela=skip_vela,
            allow_model_zoo_ref=allow_model_zoo_ref,
            isolate_source_exports=isolate_source_exports,
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
    parser.add_argument(
        "--tiers",
        default="1",
        help="comma-separated manifest tiers to include when --model is omitted or validated",
    )
    parser.add_argument("--skip-vela", action="store_true")
    parser.add_argument("--no-model-zoo-ref", action="store_true")
    parser.add_argument("--isolate-source-exports", action="store_true")
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
        isolate_source_exports=args.isolate_source_exports,
        tiers=parse_tiers(args.tiers),
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
