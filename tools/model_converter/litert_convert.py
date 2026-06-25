"""PyTorch -> LiteRT (.tflite) direct conversion via litert-torch, no ONNX.

This module implements Task 1.1 of the model-benchmark pipeline: convert a
``torch.nn.Module`` directly to a float ``.tflite`` using Google's litert-torch
(formerly ai-edge-torch) and then run int8 full-integer quantization with
ai-edge-quantizer. Neither step touches ONNX.

Two clearly separated responsibilities:

* :func:`export_float_tflite` -- the PT -> float tflite step
  (``litert_torch.convert``).
* :func:`quantize_to_int8` -- the float tflite -> int8 tflite step
  (``ai_edge_quantizer.Quantizer`` + representative calibration data).
* :func:`convert_pt_to_int8_tflite` -- the high-level pipeline that chains the
  two and reuses the ``quantize.py`` path contracts.

The "source model -> nn.Module" loader is intentionally a minimal placeholder
here (:func:`load_source_module`). Full backends (ultralytics pre-NMS head,
torchvision / timm / torch.hub / HuggingFace) are Task 1.4.

Environment note
----------------
This module is imported by the hermetic test suite (base conda, no torch). All
heavy imports (``torch``, ``litert_torch``, ``ai_edge_quantizer``,
``ai_edge_litert``) are therefore performed lazily, inside the functions that
need them, so ``import tools.model_converter.litert_convert`` always succeeds.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

# Heavy dependencies are imported lazily (see module docstring). Path helpers
# from quantize.py are pure-stdlib and safe to import unconditionally.
from tools.model_converter.quantize import artifact_dir
from tools.model_converter.quantize import float_model_path
from tools.model_converter.quantize import quantized_model_path


# --------------------------------------------------------------------------- #
# Environment bootstrap
# --------------------------------------------------------------------------- #
def _bootstrap_litert_env() -> None:
    """Apply the import-order workarounds litert-torch 0.8.0 needs on this host.

    Two problems were found while introspecting the litert-torch conda env:

    1. ``litert_torch``'s package ``__init__`` pulls in ``torchao`` ->
       ``torch._dynamo`` -> ``triton``. triton's C extension
       (``triton._C.libtriton``) segfaults inside ``create_module`` if it is
       lazily loaded *after* jax / torch_xla2 has touched the CUDA driver.
       Pre-importing ``triton`` (and ``triton.backends.compiler``) before
       ``litert_torch`` makes the C extension load cleanly.

    2. Force CPU-only JAX so torch_xla2 does not initialise a GPU context that
       would later collide with TensorFlow's eager context inside the converter.

    The bootstrap is idempotent and a no-op on hosts that already arranged the
    imports. It is only run when the heavy deps are about to be loaded.
    """
    import sys

    # Force CPU-only runtime before any converter dependency imports TensorFlow
    # or JAX. This avoids eager CUDA initialisation on hosts whose driver is
    # present but not compatible with the runtime bundled in litert-torch.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    if "triton" not in sys.modules:
        try:
            import triton  # noqa: F401
            import triton.backends.compiler  # noqa: F401
        except ImportError:
            # triton is optional for CPU-only hosts; ignore if absent.
            pass

    try:
        import jax

        jax.config.update("jax_platforms", "cpu")
    except Exception:
        # jax config is best-effort; absence must not break conversion.
        pass


# --------------------------------------------------------------------------- #
# Source model loading (placeholder -- full backends are Task 1.4)
# --------------------------------------------------------------------------- #
def load_source_module(source: dict | str, input_shape: Sequence[int]) -> tuple[Any, Any]:
    """Load a source spec into a (``nn.Module``, sample input tensor) pair.

    This is a *minimal placeholder* covering only the directly-constructed
    ``torch.nn.Module`` case, which is enough to smoke-test the conversion
    pipeline. Real source backends (ultralytics pre-NMS head, torchvision,
    timm, torch.hub, HuggingFace) are wired up in Task 1.4.

    Args:
        source: Either a ``torch.nn.Module`` instance, or a dict describing the
            source (currently only ``{"type": "module", "module": <nn.Module>}``
            is understood).
        input_shape: ``[N, C, H, W]`` shape used to build the tracing input.

    Returns:
        ``(module, sample_input)`` where ``sample_input`` is a ``torch.Tensor``.
    """
    import torch

    if isinstance(source, dict):
        if source.get("type") == "module":
            module = source["module"]
        else:
            raise NotImplementedError(
                f"source type {source.get('type')!r} is not supported yet; "
                "full source backends land in Task 1.4"
            )
    else:
        module = source

    if not hasattr(module, "forward") or not callable(getattr(module, "forward")):
        raise TypeError("source must be a torch.nn.Module (or expose .forward)")

    module.eval()
    sample_input = torch.zeros(*input_shape)
    return module, sample_input


# --------------------------------------------------------------------------- #
# Step 1: PT -> float tflite
# --------------------------------------------------------------------------- #
def export_float_tflite(
    module: Any,
    sample_args: Sequence[Any],
    output_path: str | Path,
) -> Path:
    """Convert a ``torch.nn.Module`` to a float ``.tflite`` via litert-torch.

    Args:
        module: A ``torch.nn.Module`` (already in eval mode).
        sample_args: Positional example tensors that trace the module, exactly
            as ``litert_torch.convert`` expects.
        output_path: Destination ``.tflite`` path. Parent dirs are created.

    Returns:
        The resolved ``output_path`` as a ``Path``.
    """
    _bootstrap_litert_env()
    import litert_torch

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    edge_model = litert_torch.convert(module, tuple(sample_args))
    edge_model.export(str(output_path))

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(
            f"litert_torch.convert produced an empty float tflite at {output_path}"
        )
    return output_path


# --------------------------------------------------------------------------- #
# Step 2: float tflite -> int8 tflite
# --------------------------------------------------------------------------- #
def _default_int8_recipe(quantizer: Any) -> None:
    """Configure a standard full-int8 static-range recipe on a Quantizer."""
    from ai_edge_quantizer.qtyping import TFLOperationName

    quantizer.add_static_config(
        regex=".*",
        operation_name=TFLOperationName.ALL_SUPPORTED,
        activation_num_bits=8,
        weight_num_bits=8,
    )


def _infer_signature_inputs(float_model_path: str | Path) -> tuple[str, dict]:
    """Inspect a float tflite to find (signature_name, input_details_dict).

    ai-edge-quantizer's calibration is keyed by the signature name, and the
    per-sample dicts are keyed by the signature's *input argument names*
    (e.g. ``args_0``), NOT the fully-qualified tensor names (e.g.
    ``serving_default_args_0``). The signature runner's
    ``get_input_details()`` returns a dict keyed by exactly those argument
    names, which is what callers must use to build calibration samples.
    """
    from ai_edge_litert.interpreter import Interpreter

    interp = Interpreter(model_path=str(float_model_path))
    interp.allocate_tensors()

    signatures = interp.get_signature_list()
    if signatures:
        signature_name = list(signatures)[0]
        runner = interp.get_signature_runner(signature_name)
        input_details = runner.get_input_details()  # dict keyed by arg name
    else:
        signature_name = "serving_default"
        runner = interp
        input_details = runner.get_input_details()

    return signature_name, input_details


def signature_input_names(float_model_path: str | Path) -> list[str]:
    """Return the input argument names a float tflite's signature expects.

    Calibration samples passed to :func:`quantize_to_int8` must be dicts keyed
    by these names. Exposed so callers (and Task 1.2 dataset loaders) can build
    correctly-keyed calibration data without poking the interpreter themselves.
    """
    _signature_name, input_details = _infer_signature_inputs(float_model_path)
    return list(input_details.keys())


def quantize_to_int8(
    float_model_path: str | Path,
    output_path: str | Path,
    calibration_samples: Iterable[dict[str, Any]],
    *,
    configure_recipe: Callable[[Any], None] | None = None,
) -> Path:
    """Quantize a float ``.tflite`` to int8 full-integer using ai-edge-quantizer.

    Args:
        float_model_path: Source float ``.tflite`` (e.g. from
            :func:`export_float_tflite`).
        output_path: Destination int8 ``.tflite`` path.
        calibration_samples: Iterable of ``{input_arg_name: np.ndarray}``
            dicts -- one per representative sample. The key is the signature's
            *input argument name* (e.g. ``args_0``), NOT the fully-qualified
            tensor name; use :func:`signature_input_names` to discover them.
            Array shape/dtype must match the float model's serving signature
            inputs. A minimal in-memory generator is the intended caller (real
            dataset loaders are Task 1.2).
        configure_recipe: Optional callable that receives the
            ``ai_edge_quantizer.Quantizer`` and configures the quantization
            recipe. Defaults to :func:`_default_int8_recipe` (8-bit activations
            + 8-bit weights for all supported ops, channel-wise weight
            granularity).

    Returns:
        The resolved ``output_path`` as a ``Path``.
    """
    from ai_edge_quantizer import Quantizer

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    quantizer = Quantizer(float_model_path)
    if configure_recipe is None:
        _default_int8_recipe(quantizer)
    else:
        configure_recipe(quantizer)

    if not quantizer.need_calibration:
        raise RuntimeError(
            "int8 full-integer recipe requires calibration data; got a recipe "
            "that does not need calibration."
        )

    signature_name, _input_details = _infer_signature_inputs(float_model_path)
    # ai-edge-quantizer's Calibrator iterates the sample dataset twice: once
    # internally to count ops via ``len(list(dataset))`` and once for the real
    # calibration loop. A single-pass generator would be exhausted by the count
    # pass and yield zero QSVs, so materialise to a list to stay safe.
    samples_list = list(calibration_samples)
    if not samples_list:
        raise ValueError("calibration_samples is empty; at least one sample is required.")
    calibration_data = {signature_name: samples_list}
    calibration_result = quantizer.calibrate(calibration_data)
    result = quantizer.quantize(calibration_result)
    result.export_model(str(output_path), overwrite=True)

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(
            f"ai_edge_quantizer produced an empty int8 tflite at {output_path}"
        )
    return output_path


# --------------------------------------------------------------------------- #
# High-level pipeline
# --------------------------------------------------------------------------- #
def convert_pt_to_int8_tflite(
    module: Any,
    sample_args: Sequence[Any],
    model_name: str,
    *,
    output_root: str | Path = "artifacts",
    calibration_samples: Iterable[dict[str, Any]] | None = None,
    configure_recipe: Callable[[Any], None] | None = None,
) -> dict[str, Path]:
    """Convert a ``torch.nn.Module`` all the way to an int8 ``.tflite``.

    Chains :func:`export_float_tflite` and :func:`quantize_to_int8`, writing the
    intermediate float model and the final int8 model at the paths defined by
    :func:`quantize.float_model_path` and :func:`quantize.quantized_model_path`
    under ``output_root / model_name``.

    Args:
        module: A ``torch.nn.Module`` (already in eval mode).
        sample_args: Example tensors for tracing.
        model_name: Directory/identifier under ``output_root``.
        output_root: Artifact root (mirrors the rest of the pipeline).
        calibration_samples: Representative calibration data; required for the
            int8 step. If ``None`` the float step still runs but the int8 step
            raises ``ValueError``.
        configure_recipe: Optional recipe configurator for the quantizer.

    Returns:
        ``{"float": float_path, "int8": int8_path}``.
    """
    float_path = float_model_path(model_name, output_root)
    export_float_tflite(module, sample_args, float_path)

    if calibration_samples is None:
        raise ValueError(
            "calibration_samples is required to quantize to int8; pass an "
            "iterable of {tensor_name: np.ndarray} dicts (Task 1.2 builds the "
            "real dataset loader)."
        )

    int8_path = quantized_model_path(model_name, output_root)
    quantize_to_int8(
        float_path,
        int8_path,
        calibration_samples,
        configure_recipe=configure_recipe,
    )
    return {"float": float_path, "int8": int8_path}
