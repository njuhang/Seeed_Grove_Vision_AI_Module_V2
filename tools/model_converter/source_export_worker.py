"""Crash-isolated source export worker for litert-torch conversions.

This helper runs the heavy PyTorch -> LiteRT int8 export for exactly one model
in a dedicated Python process. Parent orchestration can therefore survive
native crashes in ``litert_torch`` / TensorFlow / torch_xla2 and apply the
spec's per-model failure isolation rules instead of losing the whole batch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.model_converter import convert
from tools.model_converter.model_registry import load_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-root", default="artifacts")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    models = load_models(args.manifest)
    selected = [model for model in models if model.name == args.model]
    if len(selected) != 1:
        raise ValueError(f"expected exactly one manifest entry for {args.model}, got {len(selected)}")

    int8_path, notes = convert._export_int8_from_source(
        spec=selected[0],
        repo_root=Path(args.repo_root).resolve(),
        output_root=Path(args.output_root).resolve(),
    )
    print(json.dumps({"int8_model_path": str(int8_path), "notes": notes}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
