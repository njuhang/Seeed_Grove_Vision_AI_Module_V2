"""Hermetic tests for tools.model_converter.calibration.

Runs in the base conda env. Only numpy is required (no torch, no litert-torch).
These tests assert the *contract* Task 1.2 owes Task 1.1's
``quantize_to_int8(..., calibration_samples=...)``:

  * samples is a list of ``{input_arg_name: np.ndarray}`` dicts,
  * keyed by the float model's signature input *argument* name (e.g. "args_0"),
  * each array is float32 and shaped to the model's serving input shape.

The format compatibility with 1.1 is verified two ways here:

  1. structurally -- the loader output exactly matches the dict-of-ndarray form
     that ``quantize_to_int8`` documents (litert_convert.py L215, L225-228).
  2. by feeding loader output through the real ``quantize_to_int8`` when the
     heavy deps happen to be importable (auto-skipped in base env), otherwise
     by mock-checking the dict/ndarray shape against the form 1.1's hermetic
     smoke feeds in (``test_quantize_to_int8_passes_calibration_data_keyed_by_signature``).
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

import tools.model_converter.calibration as cal


# --------------------------------------------------------------------------- #
# Known keys
# --------------------------------------------------------------------------- #
def test_known_dataset_keys_are_supported():
    assert "coco_128" in cal.SUPPORTED_DATASETS
    assert "imagenet_1k_random" in cal.SUPPORTED_DATASETS


def test_unknown_key_raises():
    with pytest.raises(KeyError, match="unknown calibration_dataset"):
        list(cal.load_calibration_samples("does_not_exist", input_shape=[1, 3, 8, 8]))


def test_speech_commands_key_not_yet_supported():
    """Phase 6 deliverable -- must not silently produce samples now."""
    with pytest.raises(KeyError):
        list(cal.load_calibration_samples("speech_commands", input_shape=[1, 16000]))


# --------------------------------------------------------------------------- #
# Sample count / shape / dtype
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key", ["coco_128", "imagenet_1k_random"])
def test_count_shape_dtype(key):
    n = 8
    samples = list(
        cal.load_calibration_samples(key, input_shape=[1, 3, 16, 16], num_samples=n)
    )
    assert len(samples) == n
    for s in samples:
        # Contract: each sample is a dict keyed by signature input arg name.
        assert isinstance(s, dict) and len(s) >= 1
        for arg_name, arr in s.items():
            assert isinstance(arg_name, str) and arg_name  # e.g. "args_0"
            assert isinstance(arr, np.ndarray)
            # dtype matches the 1.1 contract (float tflite serving inputs).
            assert arr.dtype == np.float32
            # shape matches the requested input_shape (NCHW for these keys).
            assert list(arr.shape) == [1, 3, 16, 16]


def test_default_sample_count_is_small_and_in_quant_range():
    """Default count must be a reasonable calibration size (16-32)."""
    samples = list(cal.load_calibration_samples("coco_128", input_shape=[1, 3, 8, 8]))
    assert 16 <= len(samples) <= 32


def test_values_in_unit_range():
    """Image calibration data is normalised to [0,1] float32 (matches the
    repo's existing calibration_image_sample_data npy: min 0.0 max 1.0)."""
    samples = list(cal.load_calibration_samples("coco_128", input_shape=[1, 3, 16, 16]))
    for s in samples:
        for arr in s.values():
            assert arr.min() >= 0.0 - 1e-6
            assert arr.max() <= 1.0 + 1e-6


def test_custom_arg_name():
    """Callers may pass the float model's signature input arg name so the
    emitted dicts are keyed exactly the way quantize_to_int8 expects."""
    samples = list(
        cal.load_calibration_samples(
            "coco_128", input_shape=[1, 3, 8, 8], num_samples=2, input_name="args_0"
        )
    )
    assert list(samples[0].keys()) == ["args_0"]


# --------------------------------------------------------------------------- #
# Determinism / reproducibility (spec §7: 同输入同输出)
# --------------------------------------------------------------------------- #
def test_same_key_shape_seed_is_reproducible():
    a = list(
        cal.load_calibration_samples(
            "coco_128", input_shape=[1, 3, 16, 16], num_samples=4, seed=123
        )
    )
    b = list(
        cal.load_calibration_samples(
            "coco_128", input_shape=[1, 3, 16, 16], num_samples=4, seed=123
        )
    )
    assert len(a) == len(b)
    for sa, sb in zip(a, b):
        for k in sa:
            np.testing.assert_array_equal(sa[k], sb[k])


def test_different_seed_gives_different_samples():
    a = list(
        cal.load_calibration_samples(
            "imagenet_1k_random", input_shape=[1, 3, 16, 16], num_samples=4, seed=1
        )
    )
    b = list(
        cal.load_calibration_samples(
            "imagenet_1k_random", input_shape=[1, 3, 16, 16], num_samples=4, seed=2
        )
    )
    # At least one array must differ.
    differs = any(
        not np.array_equal(sa[k], sb[k])
        for sa, sb in zip(a, b)
        for k in sa
    )
    assert differs


def test_default_seed_is_fixed_so_default_calls_reproduce():
    """Calling without an explicit seed must still be reproducible across calls."""
    a = list(cal.load_calibration_samples("coco_128", input_shape=[1, 3, 8, 8], num_samples=3))
    b = list(cal.load_calibration_samples("coco_128", input_shape=[1, 3, 8, 8], num_samples=3))
    for sa, sb in zip(a, b):
        for k in sa:
            np.testing.assert_array_equal(sa[k], sb[k])


# --------------------------------------------------------------------------- #
# Format compatibility with Task 1.1 (quantize_to_int8)
# --------------------------------------------------------------------------- #
def test_sample_dict_form_matches_litert_convert_contract():
    """Mirror exactly the hermetic smoke in test_litert_convert:
    samples = [{"args_0": "arr0"}, ...] -- ours must be the same dict shape
    but with real float32 ndarrays of the right shape."""
    samples = list(
        cal.load_calibration_samples(
            "coco_128", input_shape=[1, 3, 8, 8], num_samples=2, input_name="args_0"
        )
    )
    # The exact form 1.1 hermetic test asserts: list of {arg_name: ndarray}.
    assert all(list(s.keys()) == ["args_0"] for s in samples)
    assert all(isinstance(s["args_0"], np.ndarray) for s in samples)


def test_real_quantize_to_int8_accepts_loader_output(tmp_path):
    """If torch / litert_torch / ai_edge_quantizer are importable (litert-torch
    conda env), run the loader's output through the real quantize_to_int8 and
    assert it produces a non-empty int8 tflite. Otherwise skip -- the
    structural assertions above already lock the contract."""
    pytest.importorskip("torch")
    pytest.importorskip("litert_torch")
    pytest.importorskip("ai_edge_quantizer")

    import torch
    import torch.nn as nn

    from tools.model_converter import litert_convert as lc

    class TinyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.relu(self.conv(x))

    model = TinyNet().eval()
    sample = torch.randn(1, 3, 8, 8)
    float_path = tmp_path / "tiny_float.tflite"
    lc.export_float_tflite(model, (sample,), float_path)

    input_names = lc.signature_input_names(float_path)
    samples = list(
        cal.load_calibration_samples(
            "coco_128",
            input_shape=[1, 3, 8, 8],
            num_samples=4,
            input_name=input_names[0],
        )
    )
    int8_path = tmp_path_for_real_smoke() / "tiny_int8.tflite"
    lc.quantize_to_int8(float_path, int8_path, samples)
    assert int8_path.stat().st_size > 0


def tmp_path_for_real_smoke():
    import tempfile
    return next(_tmp_roots())


def _tmp_roots():
    yield _make_tmp()


def _make_tmp():
    import tempfile
    from pathlib import Path
    d = Path(tempfile.mkdtemp(prefix="calib_real_"))
    return d
