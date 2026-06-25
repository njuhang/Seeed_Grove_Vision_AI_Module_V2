"""Representative calibration data loaders for litert-torch int8 quantization.

This is Task 1.2 of the model-benchmark pipeline. It produces representative
samples for the ``calibration_samples`` argument of
:func:`tools.model_converter.litert_convert.quantize_to_int8`.

Sample format contract (must match Task 1.1 exactly)
----------------------------------------------------
``quantize_to_int8(..., calibration_samples=...)`` (litert_convert.py L215,
L225-228) expects:

    calibration_samples: Iterable[dict[str, np.ndarray]]

i.e. an iterable of dicts, **one per sample**, each mapping the float model's
signature input *argument name* (e.g. ``"args_0"`` -- NOT the fully-qualified
tensor name like ``"serving_default_args_0"``) to a ``np.ndarray`` whose
``shape`` and ``dtype`` match the float model's serving signature inputs.

The hermetic smoke in ``test_litert_convert.py`` builds samples of exactly this
form::

    samples = [{"args_0": "arr0"}, {"args_0": "arr1"}]

and the real smoke feeds ``np.random.rand(1, 3, 8, 8).astype(np.float32)``.
:func:`load_calibration_samples` therefore yields dicts of the same shape,
keyed by the caller-supplied ``input_name`` (default ``"args_0"``), with
float32 arrays normalised to ``[0, 1]`` and shaped to the model's
``input_shape``.

Design (spec §7: offline, deterministic, reproducible)
------------------------------------------------------
* **No network.** Nothing is downloaded. Samples are generated in-memory from a
  fixed seed, so the same ``(key, input_shape, seed)`` triple always produces
  byte-identical arrays (``np.random.default_rng`` is deterministic and
  seed-stable across processes).
* **Deterministic.** The default seed is a module constant; every call without
  an explicit ``seed`` reproduces the same samples.
* **Small.** Default ``num_samples`` is 32 -- a reasonable calibration set
  (16-32 per spec) that keeps the in-memory footprint tiny.
* ``coco_128`` (OD) and ``imagenet_1k_random`` (classification) both produce
  ``[0,1]`` float32 image tensors. ``speech_commands`` produces deterministic
  MFCC-like float32 feature tensors for KWS models.
"""

from __future__ import annotations

from typing import Iterator, Sequence

import numpy as np

# --------------------------------------------------------------------------- #
# Public constants
# --------------------------------------------------------------------------- #
#: Calibration keys supported by the current conversion flow.
SUPPORTED_DATASETS: frozenset[str] = frozenset({"coco_128", "imagenet_1k_random", "speech_commands"})

#: The default signature input argument name used when the caller does not pass
#: one explicitly. Matches the name ai-edge-torch/litert-torch emit for the
#: single-positional-input trace (``args_0``); callers with a different
#: signature should pass ``input_name`` (discovered via
#: ``litert_convert.signature_input_names``).
DEFAULT_INPUT_NAME: str = "args_0"

#: Default sample count. Spec §7 asks for a small but quantization-adequate
#: representative set; 32 sits at the upper end of the 16-32 range.
DEFAULT_NUM_SAMPLES: int = 32

#: Fixed default seed so calls without an explicit seed are reproducible.
DEFAULT_SEED: int = 0xC0C0  # 49344


# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #
def load_calibration_samples(
    calibration_dataset: str,
    *,
    input_shape: Sequence[int],
    num_samples: int | None = None,
    input_name: str = DEFAULT_INPUT_NAME,
    seed: int | None = None,
) -> Iterator[dict[str, np.ndarray]]:
    """Yield representative calibration samples for int8 quantization.

    Each yielded item is a ``{input_name: np.ndarray}`` dict whose array is
    ``float32`` and shaped to ``input_shape`` -- exactly the form
    :func:`litert_convert.quantize_to_int8` consumes (litert_convert.py L215,
    L225-228).

    Args:
        calibration_dataset: Key declared under ``calibration_dataset`` in
            ``configs/models.yaml``. ``coco_128`` and ``imagenet_1k_random``
            produce image-like [0,1] tensors; ``speech_commands`` produces
            MFCC-like feature tensors for audio/KWS calibration.
        input_shape: Shape of one model input sample, typically
            ``[N, C, H, W]`` (e.g. ``[1, 3, 192, 192]``). Arrays are produced at
            exactly this shape, so callers pass the float model's serving input
            shape verbatim.
        num_samples: Number of representative samples to yield. ``None`` falls
            back to :data:`DEFAULT_NUM_SAMPLES` (32). Calibration needs only a
            small set; keep it in the 16-32 range.
        input_name: Signature input *argument name* the produced dicts are
            keyed by. Defaults to :data:`DEFAULT_INPUT_NAME` (``"args_0"``);
            pass the name returned by
            :func:`litert_convert.signature_input_names` to match an arbitrary
            float model exactly.
        seed: PRNG seed for deterministic generation. ``None`` uses
            :data:`DEFAULT_SEED` so default calls are reproducible across
            processes and runs.

    Yields:
        ``{input_name: np.ndarray}`` dicts, one per sample.

    Raises:
        KeyError: If ``calibration_dataset`` is not in :data:`SUPPORTED_DATASETS`
            (covers unknown keys).
    """
    if calibration_dataset not in SUPPORTED_DATASETS:
        raise KeyError(
            f"unknown calibration_dataset {calibration_dataset!r}; "
            f"supported keys: {sorted(SUPPORTED_DATASETS)} "
            ""
        )

    if num_samples is None:
        num_samples = DEFAULT_NUM_SAMPLES
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")

    # Per-key seed offset so different dataset keys with the same numeric seed
    # produce genuinely different arrays (otherwise coco_128 and
    # imagenet_1k_random with seed=1 would be identical, which would be
    # misleading for a per-dataset loader). The offset is a stable hash of the
    # key, not Python's randomized hash(), so it reproduces across processes.
    effective_seed = (seed if seed is not None else DEFAULT_SEED) + _stable_key_offset(
        calibration_dataset
    )

    rng = np.random.default_rng(effective_seed)
    shape = tuple(int(d) for d in input_shape)

    for _ in range(num_samples):
        if calibration_dataset == "speech_commands":
            arr = rng.normal(loc=0.0, scale=1.0, size=shape).astype(np.float32)
            np.clip(arr, -4.0, 4.0, out=arr)
        else:
            arr = rng.random(size=shape, dtype=np.float32)
            # rng.random already covers [0,1); clip away the rare 1.0 boundary
            # so the [0,1] guarantee for image keys is exact for tests.
            np.clip(arr, 0.0, 1.0, out=arr)
        yield {input_name: arr}


def _stable_key_offset(key: str) -> int:
    """Return a process-stable integer derived from ``key``.

    Built-in ``hash(str)`` is randomized per process (PYTHONHASHSEED), which
    would break cross-process reproducibility of the per-key seed offset. A
    simple sum of byte values is deterministic and good enough to keep
    different dataset keys' streams from colliding.
    """
    return sum(key.encode("utf-8"))


# --------------------------------------------------------------------------- #
# Convenience: support check for callers
# --------------------------------------------------------------------------- #
def is_supported(calibration_dataset: str) -> bool:
    """Return True iff ``calibration_dataset`` is loadable this phase."""
    return calibration_dataset in SUPPORTED_DATASETS
