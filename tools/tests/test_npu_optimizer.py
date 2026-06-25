import json
from pathlib import Path
from unittest.mock import patch

from tools.model_converter.npu_optimizer import OptimizerCandidate
from tools.model_converter.npu_optimizer import NpuOptimizerConfig
from tools.model_converter.npu_optimizer import optimize_tflite_for_npu
from tools.model_converter.npu_optimizer import should_optimize_for_npu
from tools.model_converter.tflite_rewrite import NpuRewriteRule
from tools.model_converter.tflite_rewrite import apply_npu_rewrite_rules


def _write_vela_sidecar(path: Path, *, cpu_ops: int, npu_ops: int, fallback: list[str]) -> None:
    total = cpu_ops + npu_ops
    util = 0.0 if total == 0 else round((npu_ops / total) * 100, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "cpu_ops": cpu_ops,
                "npu_ops": npu_ops,
                "npu_utilization_pct": util,
                "cpu_fallback_ops": fallback,
            }
        ),
        encoding="utf-8",
    )


def test_should_optimize_for_low_npu_or_real_fallback() -> None:
    assert should_optimize_for_npu({"npu_utilization_pct": 94.9, "cpu_fallback_ops": ["Passthrough"]})
    assert should_optimize_for_npu({"npu_utilization_pct": 99.0, "cpu_fallback_ops": ["Transpose"]})
    assert not should_optimize_for_npu({"npu_utilization_pct": 99.0, "cpu_fallback_ops": ["Passthrough"]})
    assert not should_optimize_for_npu({"npu_utilization_pct": 100.0, "cpu_fallback_ops": []})


def test_apply_npu_rewrite_rules_runs_registered_rules_in_order(tmp_path: Path) -> None:
    source = tmp_path / "source.tflite"
    output = tmp_path / "output.tflite"
    source.write_bytes(b"base")

    def first(input_path: Path, output_path: Path) -> int:
        output_path.write_bytes(input_path.read_bytes() + b"+first")
        return 1

    def second(input_path: Path, output_path: Path) -> int:
        output_path.write_bytes(input_path.read_bytes() + b"+second")
        return 2

    applied = apply_npu_rewrite_rules(
        source,
        output,
        rules=(
            NpuRewriteRule("first", "first rewrite", first),
            NpuRewriteRule("second", "second rewrite", second),
        ),
    )

    assert output.read_bytes() == b"base+first+second"
    assert [(rule.name, rule.rewritten_count) for rule in applied] == [("first", 1), ("second", 2)]


def test_optimize_tflite_for_npu_selects_best_candidate_and_writes_trace(tmp_path: Path) -> None:
    int8_path = tmp_path / "model_int8.tflite"
    int8_path.write_bytes(b"int8")
    baseline_info_path = tmp_path / "vela" / "model.vela_info.json"
    _write_vela_sidecar(baseline_info_path, cpu_ops=4, npu_ops=4, fallback=["Transpose"])
    baseline_vela_path = tmp_path / "model_vela.tflite"
    baseline_vela_path.write_bytes(b"baseline-vela")
    final_vela_path = tmp_path / "final" / "model_vela.tflite"

    candidates = [
        OptimizerCandidate(name="weak", rules=("weak",)),
        OptimizerCandidate(name="strong", rules=("strong",)),
    ]

    def fake_apply(input_path: Path, output_path: Path, *, rules=None, rule_names=None):
        output_path.write_bytes(input_path.read_bytes() + b"+" + ",".join(rule_names or ()).encode())
        return []

    def fake_write_vela_artifacts(*, model_name, model_path, output_dir, **kwargs):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        generated = output_dir / f"{Path(model_path).stem}_vela.tflite"
        generated.write_bytes(f"{model_name}:{Path(model_path).name}".encode())
        sidecar = output_dir / f"{model_name}.vela_info.json"
        if model_name.endswith("weak"):
            _write_vela_sidecar(sidecar, cpu_ops=2, npu_ops=8, fallback=["Passthrough"])
        else:
            _write_vela_sidecar(sidecar, cpu_ops=0, npu_ops=10, fallback=[])
        return sidecar

    with (
        patch("tools.model_converter.npu_optimizer.apply_npu_rewrite_rules", side_effect=fake_apply),
        patch("tools.model_converter.npu_optimizer.write_vela_artifacts", side_effect=fake_write_vela_artifacts),
    ):
        result = optimize_tflite_for_npu(
            model_name="model",
            int8_path=int8_path,
            baseline_vela_path=baseline_vela_path,
            baseline_info_path=baseline_info_path,
            final_vela_path=final_vela_path,
            artifact_dir=tmp_path / "model",
            vela_dir=tmp_path / "vela",
            config=NpuOptimizerConfig(candidates=tuple(candidates)),
        )

    assert result.applied is True
    assert result.best_candidate == "strong"
    assert result.optimized_vela_path == final_vela_path
    assert final_vela_path.read_bytes().startswith(b"model_npu_opt_strong")

    trace = json.loads((tmp_path / "model" / "optimizer_trace.json").read_text(encoding="utf-8"))
    assert trace["triggered"] is True
    assert trace["best_candidate"] == "strong"
    assert trace["baseline"]["npu_utilization_pct"] == 50.0
    assert trace["optimized"]["npu_utilization_pct"] == 100.0
    assert trace["board_verification"]["required"] is True
    assert trace["board_verification"]["status"] == "pending"


def test_optimize_tflite_for_npu_skips_high_npu_passthrough_only_model(tmp_path: Path) -> None:
    int8_path = tmp_path / "model_int8.tflite"
    int8_path.write_bytes(b"int8")
    baseline_info_path = tmp_path / "vela" / "model.vela_info.json"
    _write_vela_sidecar(baseline_info_path, cpu_ops=1, npu_ops=99, fallback=["Passthrough"])
    baseline_vela_path = tmp_path / "baseline_vela.tflite"
    baseline_vela_path.write_bytes(b"baseline")
    final_vela_path = tmp_path / "final_vela.tflite"

    result = optimize_tflite_for_npu(
        model_name="model",
        int8_path=int8_path,
        baseline_vela_path=baseline_vela_path,
        baseline_info_path=baseline_info_path,
        final_vela_path=final_vela_path,
        artifact_dir=tmp_path / "model",
        vela_dir=tmp_path / "vela",
    )

    assert result.applied is False
    assert result.best_candidate is None
    assert not (tmp_path / "model" / "optimizer_trace.json").exists()
