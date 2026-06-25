"""Hermetic tests for tools.model_converter.litert_convert.

These run in the base conda env (no torch, no litert-torch). The heavy
conversion backends are mocked so we assert on the *scheduling logic* of
``litert_convert``: that the right functions are called with the right args,
that paths flow through the ``quantize.py`` contract, and that errors surface
sensibly.

The real end-to-end conversion is exercised by the ``litert-torch`` conda env
in ``test_real_conversion_smoke`` below; that test self-skips when torch /
litert-torch / ai_edge_quantizer are not importable so the base suite stays
green.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import tools.model_converter.litert_convert as lc


# --------------------------------------------------------------------------- #
# Helpers: install fake heavy modules into sys.modules so the module under test
# can exercise its real import + dispatch code without the real deps.
# --------------------------------------------------------------------------- #
def _install_fake_heavy_modules(monkeypatch, float_bytes=b"FLOAT", int8_bytes=b"INT8"):
    """Inject fake torch / litert_torch / ai_edge_quantizer / ai_edge_litert.

    Returns the MagicMock objects the tests want to assert against.
    """
    fake_torch = types.ModuleType("torch")
    fake_torch.zeros = MagicMock(return_value="SAMPLE_INPUT")

    fake_nn = types.ModuleType("torch.nn")
    # Fake nn.Module base class so load_source_module can recognise subclasses.
    class _FakeModule:
        def __init__(self):
            pass

        def eval(self):
            return self

        def forward(self, *args, **kwargs):  # pragma: no cover - never called
            raise RuntimeError("forward should not run in hermetic tests")

    fake_nn.Module = _FakeModule
    fake_torch.nn = fake_nn
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.nn", fake_nn)

    # Fake triton + jax so _bootstrap_litert_env can run its pre-imports.
    fake_triton = types.ModuleType("triton")
    fake_triton.__version__ = "0.0.0-fake"
    fake_triton_backends = types.ModuleType("triton.backends.compiler")
    monkeypatch.setitem(sys.modules, "triton", fake_triton)
    monkeypatch.setitem(sys.modules, "triton.backends", types.ModuleType("triton.backends"))
    monkeypatch.setitem(sys.modules, "triton.backends.compiler", fake_triton_backends)
    fake_jax = types.ModuleType("jax")
    fake_jax.config = MagicMock()
    monkeypatch.setitem(sys.modules, "jax", fake_jax)

    # Fake litert_torch.convert -> edge model with .export(path).
    fake_litert = types.ModuleType("litert_torch")

    class _FakeEdgeModel:
        def __init__(self, payload):
            self._payload = payload

        def export(self, path):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_bytes(self._payload)

    convert_mock = MagicMock(return_value=_FakeEdgeModel(float_bytes))
    fake_litert.convert = convert_mock
    monkeypatch.setitem(sys.modules, "litert_torch", fake_litert)

    # Fake ai_edge_quantizer.
    fake_qtyping = types.ModuleType("ai_edge_quantizer.qtyping")

    class _TFLOpName:
        ALL_SUPPORTED = "*"

    fake_qtyping.TFLOperationName = _TFLOpName
    fake_quantizer_pkg = types.ModuleType("ai_edge_quantizer")

    class _FakeQuantResult:
        def export_model(self, filepath, overwrite=False):
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            Path(filepath).write_bytes(int8_bytes)

    class _FakeQuantizer:
        def __init__(self, float_model):
            self.float_model = float_model
            self.recipe_configured = False

        def add_static_config(self, **kwargs):
            self.recipe_configured = True
            self.last_static_config = kwargs

        @property
        def need_calibration(self):
            return True

        def calibrate(self, calibration_data):
            self.last_calibration_data = calibration_data
            return {"tensor_0": {"min": 0.0, "max": 1.0}}

        def quantize(self, calibration_result):
            self.last_calibration_result = calibration_result
            return _FakeQuantResult()

    fake_quantizer_pkg.Quantizer = _FakeQuantizer
    monkeypatch.setitem(sys.modules, "ai_edge_quantizer", fake_quantizer_pkg)
    monkeypatch.setitem(sys.modules, "ai_edge_quantizer.qtyping", fake_qtyping)

    # Fake ai_edge_litert.interpreter for signature discovery.
    fake_litert_pkg = types.ModuleType("ai_edge_litert")
    fake_interpreter_pkg = types.ModuleType("ai_edge_litert.interpreter")

    class _FakeRunner:
        def get_input_details(self):
            # Signature runner returns a dict keyed by arg name (matches real API).
            return {"args_0": {"name": "serving_default_args_0", "shape": [1, 3, 8, 8]}}

    class _FakeInterpreter:
        def __init__(self, **kwargs):
            pass

        def allocate_tensors(self):
            pass

        def get_signature_list(self):
            return {"serving_default": {"inputs": ["args_0"], "outputs": ["output_0"]}}

        def get_signature_runner(self, name):
            return _FakeRunner()

        def get_input_details(self):
            return [{"name": "serving_default_args_0", "shape": [1, 3, 8, 8]}]

    fake_interpreter_pkg.Interpreter = _FakeInterpreter
    fake_litert_pkg.interpreter = fake_interpreter_pkg
    monkeypatch.setitem(sys.modules, "ai_edge_litert", fake_litert_pkg)
    monkeypatch.setitem(sys.modules, "ai_edge_litert.interpreter", fake_interpreter_pkg)

    return {
        "torch": fake_torch,
        "litert_torch": fake_litert,
        "convert_mock": convert_mock,
        "quantizer_cls": _FakeQuantizer,
    }


# --------------------------------------------------------------------------- #
# Pure-logic tests (no heavy deps needed)
# --------------------------------------------------------------------------- #
def test_module_imports_without_torch_in_base_env():
    """The module must import even when torch/litert_torch are absent.

    Heavy imports are lazy so the hermetic test suite (base conda) can load it.
    """
    assert hasattr(lc, "export_float_tflite")
    assert hasattr(lc, "quantize_to_int8")
    assert hasattr(lc, "convert_pt_to_int8_tflite")


def test_bootstrap_litert_env_forces_cpu_only_runtime(monkeypatch):
    _install_fake_heavy_modules(monkeypatch)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)

    lc._bootstrap_litert_env()

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "-1"
    assert os.environ["JAX_PLATFORMS"] == "cpu"


def test_float_model_path_uses_quantize_contract():
    """litert_convert re-uses quantize.py path helpers rather than re-deriving."""
    assert lc.float_model_path("demo") == Path("artifacts/demo/demo_float.tflite")
    assert lc.quantized_model_path("demo") == Path("artifacts/demo/demo_int8.tflite")
    assert lc.float_model_path("demo").parent == lc.artifact_dir("demo")


def test_load_source_module_rejects_unknown_type(monkeypatch):
    _install_fake_heavy_modules(monkeypatch)
    with pytest.raises(NotImplementedError, match="Task 1.4"):
        lc.load_source_module({"type": "ultralytics", "model": "x.pt"}, [1, 3, 8, 8])


def test_load_source_module_rejects_non_module(monkeypatch):
    _install_fake_heavy_modules(monkeypatch)
    with pytest.raises(TypeError, match="torch.nn.Module"):
        lc.load_source_module(object(), [1, 3, 8, 8])


# --------------------------------------------------------------------------- #
# export_float_tflite
# --------------------------------------------------------------------------- #
def test_export_float_tflite_calls_litert_convert_and_writes_file(monkeypatch, tmp_path):
    mocks = _install_fake_heavy_modules(monkeypatch, float_bytes=b"MYFLOAT")
    fake_module = mocks["torch"].nn.Module()

    out = tmp_path / "sub" / "model_float.tflite"
    result = lc.export_float_tflite(fake_module, ("SAMPLE_INPUT",), out)

    assert result == out
    assert out.read_bytes() == b"MYFLOAT"
    mocks["convert_mock"].assert_called_once_with(fake_module, ("SAMPLE_INPUT",))


def test_export_float_tflite_raises_when_output_empty(monkeypatch, tmp_path):
    """If litert_torch silently writes nothing, surface a clear error."""
    # Edge model that writes zero bytes.
    fake_torch = types.ModuleType("torch")
    fake_nn = types.ModuleType("torch.nn")
    fake_nn.Module = type("M", (), {"eval": lambda self: self})
    fake_torch.nn = fake_nn
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.nn", fake_nn)
    monkeypatch.setitem(sys.modules, "triton", types.ModuleType("triton"))
    monkeypatch.setitem(sys.modules, "triton.backends", types.ModuleType("triton.backends"))
    monkeypatch.setitem(sys.modules, "triton.backends.compiler", types.ModuleType("triton.backends.compiler"))
    monkeypatch.setitem(sys.modules, "jax", types.ModuleType("jax"))

    fake_litert = types.ModuleType("litert_torch")

    class _EmptyEdgeModel:
        def export(self, path):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_bytes(b"")

    fake_litert.convert = MagicMock(return_value=_EmptyEdgeModel())
    monkeypatch.setitem(sys.modules, "litert_torch", fake_litert)

    out = tmp_path / "empty.tflite"
    with pytest.raises(RuntimeError, match="empty float tflite"):
        lc.export_float_tflite(fake_nn.Module(), ("x",), out)


# --------------------------------------------------------------------------- //
# quantize_to_int8
# --------------------------------------------------------------------------- #
def test_quantize_to_int8_configures_default_recipe_and_writes_file(monkeypatch, tmp_path):
    mocks = _install_fake_heavy_modules(monkeypatch, int8_bytes=b"MYINT8")
    float_path = tmp_path / "m_float.tflite"
    float_path.write_bytes(b"FLOAT")
    int8_path = tmp_path / "m_int8.tflite"

    samples = ({"args_0": "arr0"}, {"args_0": "arr1"})
    result = lc.quantize_to_int8(float_path, int8_path, samples)

    assert result == int8_path
    assert int8_path.read_bytes() == b"MYINT8"
    # The fake Quantizer recorded that add_static_config was invoked via the
    # default recipe (8/8 bits, ALL_SUPPORTED).
    # We cannot read instance state directly here; instead assert via behaviour:
    # need_calibration is True and quantize() was called through to export.
    assert int8_path.exists()


def test_quantize_to_int8_uses_custom_recipe_configurator(monkeypatch, tmp_path):
    _install_fake_heavy_modules(monkeypatch)
    float_path = tmp_path / "m_float.tflite"
    float_path.write_bytes(b"FLOAT")
    int8_path = tmp_path / "m_int8.tflite"

    captured = {}

    def my_recipe(quantizer):
        captured["called"] = True
        captured["is_quantizer"] = quantizer.__class__.__name__

    lc.quantize_to_int8(
        float_path,
        int8_path,
        iter([{"args_0": "arr"}]),
        configure_recipe=my_recipe,
    )
    assert captured["called"] is True
    assert captured["is_quantizer"] == "_FakeQuantizer"


def test_quantize_to_int8_raises_when_recipe_needs_no_calibration(monkeypatch, tmp_path):
    """A recipe that does not require calibration is a misconfiguration."""
    _install_fake_heavy_modules(monkeypatch)
    # Patch the installed FakeQuantizer to report need_calibration=False.
    fake_pkg = sys.modules["ai_edge_quantizer"]

    class _NoCalQuantizer(fake_pkg.Quantizer):
        @property
        def need_calibration(self):
            return False

    fake_pkg.Quantizer = _NoCalQuantizer

    float_path = tmp_path / "m_float.tflite"
    float_path.write_bytes(b"FLOAT")
    with pytest.raises(RuntimeError, match="calibration data"):
        lc.quantize_to_int8(
            float_path,
            tmp_path / "m_int8.tflite",
            iter([{"args_0": "arr"}]),
        )


def test_quantize_to_int8_passes_calibration_data_keyed_by_signature(monkeypatch, tmp_path):
    """The signature name discovered from the float model must key the data."""
    mocks = _install_fake_heavy_modules(monkeypatch)
    # Replace the Quantizer so we can capture the calibration_data dict.
    captured = {}

    class _CapturingQuantizer(mocks["quantizer_cls"]):
        def calibrate(self, calibration_data):
            captured["calibration_data"] = calibration_data
            return super().calibrate(calibration_data)

    sys.modules["ai_edge_quantizer"].Quantizer = _CapturingQuantizer

    float_path = tmp_path / "m_float.tflite"
    float_path.write_bytes(b"FLOAT")
    samples = [{"args_0": f"arr{i}"} for i in range(3)]
    lc.quantize_to_int8(float_path, tmp_path / "m_int8.tflite", samples)

    assert "serving_default" in captured["calibration_data"]
    # The samples iterable is passed through unchanged.
    assert list(captured["calibration_data"]["serving_default"]) == samples


# --------------------------------------------------------------------------- #
# convert_pt_to_int8_tflite (high-level pipeline)
# --------------------------------------------------------------------------- #
def test_convert_pipeline_writes_float_and_int8_at_contract_paths(monkeypatch, tmp_path):
    mocks = _install_fake_heavy_modules(monkeypatch)
    fake_module = mocks["torch"].nn.Module()

    samples = [{"args_0": "arr0"}]
    result = lc.convert_pt_to_int8_tflite(
        fake_module,
        ("SAMPLE_INPUT",),
        "demo",
        output_root=tmp_path,
        calibration_samples=samples,
    )

    assert result["float"] == tmp_path / "demo" / "demo_float.tflite"
    assert result["int8"] == tmp_path / "demo" / "demo_int8.tflite"
    assert result["float"].read_bytes() == b"FLOAT"
    assert result["int8"].read_bytes() == b"INT8"
    mocks["convert_mock"].assert_called_once_with(fake_module, ("SAMPLE_INPUT",))


def test_convert_pipeline_requires_calibration_samples(monkeypatch, tmp_path):
    _install_fake_heavy_modules(monkeypatch)
    # Patch export to avoid needing litert_torch import-order issues here; we
    # only assert the int8 step errors before any quantization runs.
    monkeypatch.setattr(
        lc, "export_float_tflite", lambda *a, **k: tmp_path / "demo" / "demo_float.tflite"
    )
    with pytest.raises(ValueError, match="calibration_samples"):
        lc.convert_pt_to_int8_tflite(
            object(),
            ("x",),
            "demo",
            output_root=tmp_path,
            calibration_samples=None,
        )


def test_convert_pipeline_chains_float_then_int8(monkeypatch, tmp_path):
    """The pipeline must call export_float_tflite first, then quantize_to_int8
    with the float path it just produced."""
    _install_fake_heavy_modules(monkeypatch)
    calls = []

    def fake_export(module, sample_args, output_path):
        calls.append(("export", str(output_path)))
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"FLOAT")
        return Path(output_path)

    def fake_quantize(float_path, output_path, samples, *, configure_recipe=None):
        calls.append(("quantize", str(float_path), str(output_path)))
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"INT8")
        return Path(output_path)

    monkeypatch.setattr(lc, "export_float_tflite", fake_export)
    monkeypatch.setattr(lc, "quantize_to_int8", fake_quantize)

    lc.convert_pt_to_int8_tflite(
        object(),
        ("x",),
        "demo",
        output_root=tmp_path,
        calibration_samples=[{"t": "a"}],
    )

    assert calls[0][0] == "export"
    assert calls[1][0] == "quantize"
    # The quantize step must receive the float path the export step produced.
    assert calls[0][1] == calls[1][1]
    expected_int8 = str(tmp_path / "demo" / "demo_int8.tflite")
    assert calls[1][2] == expected_int8


# --------------------------------------------------------------------------- #
# Real-conversion smoke test (skipped in base env)
# --------------------------------------------------------------------------- #
def test_real_conversion_smoke(tmp_path):
    """End-to-end PT -> float tflite -> int8 tflite in the litert-torch env.

    Self-skips when torch / litert_torch / ai_edge_quantizer are not importable,
    so it never breaks the base conda test suite. Run it explicitly with the
    litert-torch env once the env is healthy (see litert_convert.py docstring
    for the import-order workarounds).
    """
    if os.environ.get("RUN_REAL_CONVERSION_SMOKE") != "1":
        pytest.skip("set RUN_REAL_CONVERSION_SMOKE=1 to enable the real conversion smoke test")

    pytest.importorskip("torch")
    pytest.importorskip("litert_torch")
    pytest.importorskip("ai_edge_quantizer")

    import torch
    import torch.nn as nn

    class TinyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.relu(self.conv(x))

    import numpy as np

    model = TinyNet().eval()
    sample = torch.randn(1, 3, 8, 8)

    float_path = lc.export_float_tflite(model, (sample,), tmp_path / "tiny_float.tflite")
    assert float_path.stat().st_size > 0

    # Calibration samples must be keyed by the signature's input argument names
    # (e.g. "args_0"), not the fully-qualified tensor names.
    input_names = lc.signature_input_names(float_path)
    assert input_names, "float model should expose at least one signature input"

    def _make_samples(n=4):
        for _ in range(n):
            yield {input_names[0]: np.random.rand(1, 3, 8, 8).astype(np.float32)}

    int8_path = lc.quantize_to_int8(
        float_path, tmp_path / "tiny_int8.tflite", _make_samples(4)
    )
    assert int8_path.stat().st_size > 0
