import json
import shutil
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

from tools.model_converter.tflite_rewrite import apply_npu_rewrite_rules
from tools.model_converter.vela_compile import write_vela_artifacts


NPU_OPTIMIZATION_THRESHOLD = 95.0


@dataclass(frozen=True)
class OptimizerCandidate:
    name: str
    rules: tuple[str, ...]


@dataclass(frozen=True)
class NpuOptimizerConfig:
    npu_threshold_pct: float = NPU_OPTIMIZATION_THRESHOLD
    candidates: tuple[OptimizerCandidate, ...] = field(
        default_factory=lambda: (
            OptimizerCandidate(
                name="all_safe_rewrites",
                rules=(
                    "float16_dequantize_constants",
                    "int8_conv_biases_to_int32",
                    "prelu_float_islands_to_int8",
                    "serving_default_signature",
                ),
            ),
        )
    )


@dataclass(frozen=True)
class NpuOptimizationResult:
    applied: bool
    triggered: bool
    reason: str
    best_candidate: str | None
    baseline_info_path: Path | None
    optimized_info_path: Path | None
    baseline_vela_path: Path | None
    optimized_vela_path: Path | None
    trace_path: Path | None
    notes: list[str]
    baseline_summary: dict[str, Any]
    optimized_summary: dict[str, Any] | None

    def to_summary(self) -> dict[str, Any]:
        return {
            "applied": self.applied,
            "triggered": self.triggered,
            "reason": self.reason,
            "best_candidate": self.best_candidate,
            "baseline_info_path": _path_or_none(self.baseline_info_path),
            "optimized_info_path": _path_or_none(self.optimized_info_path),
            "baseline_vela_path": _path_or_none(self.baseline_vela_path),
            "optimized_vela_path": _path_or_none(self.optimized_vela_path),
            "trace_path": _path_or_none(self.trace_path),
            "notes": list(self.notes),
            "baseline_summary": dict(self.baseline_summary),
            "optimized_summary": dict(self.optimized_summary or {}),
        }


def should_optimize_for_npu(
    vela_summary: dict[str, Any],
    *,
    threshold_pct: float = NPU_OPTIMIZATION_THRESHOLD,
) -> bool:
    npu_pct = _float_value(vela_summary.get("npu_utilization_pct"), default=0.0)
    fallback_ops = list(vela_summary.get("cpu_fallback_ops") or [])
    real_fallback_ops = [op for op in fallback_ops if op != "Passthrough"]
    return npu_pct < threshold_pct or bool(real_fallback_ops)


def optimize_tflite_for_npu(
    *,
    model_name: str,
    int8_path: str | Path,
    baseline_vela_path: str | Path | None,
    baseline_info_path: str | Path | None,
    final_vela_path: str | Path,
    artifact_dir: str | Path,
    vela_dir: str | Path,
    config: NpuOptimizerConfig | None = None,
    vela_kwargs: dict[str, Any] | None = None,
) -> NpuOptimizationResult:
    config = config or NpuOptimizerConfig()
    int8_path = Path(int8_path)
    final_vela_path = Path(final_vela_path)
    artifact_dir = Path(artifact_dir)
    vela_dir = Path(vela_dir)
    baseline_vela = Path(baseline_vela_path) if baseline_vela_path else None
    baseline_info = Path(baseline_info_path) if baseline_info_path else None
    vela_kwargs = vela_kwargs or {}

    if baseline_info is None or not baseline_info.exists():
        return NpuOptimizationResult(
            applied=False,
            triggered=False,
            reason="missing baseline vela_info",
            best_candidate=None,
            baseline_info_path=baseline_info,
            optimized_info_path=None,
            baseline_vela_path=baseline_vela,
            optimized_vela_path=None,
            trace_path=None,
            notes=["npu optimizer skipped: missing baseline vela_info"],
            baseline_summary={},
            optimized_summary=None,
        )

    baseline_summary = _load_json(baseline_info)
    if not should_optimize_for_npu(
        baseline_summary,
        threshold_pct=config.npu_threshold_pct,
    ):
        return NpuOptimizationResult(
            applied=False,
            triggered=False,
            reason="baseline already satisfies NPU optimization threshold",
            best_candidate=None,
            baseline_info_path=baseline_info,
            optimized_info_path=None,
            baseline_vela_path=baseline_vela,
            optimized_vela_path=None,
            trace_path=None,
            notes=["npu optimizer skipped: baseline already high-NPU"],
            baseline_summary=baseline_summary,
            optimized_summary=None,
        )

    artifact_dir.mkdir(parents=True, exist_ok=True)
    trace_path = artifact_dir / "optimizer_trace.json"
    candidate_results: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    for candidate in config.candidates:
        candidate_model = artifact_dir / f"{model_name}_npu_opt_{candidate.name}.tflite"
        rewrite_results = apply_npu_rewrite_rules(
            int8_path,
            candidate_model,
            rule_names=candidate.rules,
        )
        candidate_model_name = f"{model_name}_npu_opt_{candidate.name}"
        candidate_info_path = write_vela_artifacts(
            model_name=candidate_model_name,
            model_path=candidate_model,
            output_dir=vela_dir,
            **vela_kwargs,
        )
        candidate_summary = _load_json(candidate_info_path)
        generated_vela_path = _find_generated_vela_model(candidate_model, vela_dir)
        entry = {
            "name": candidate.name,
            "rules": [
                {
                    "name": rule.name,
                    "description": rule.description,
                    "rewritten_count": rule.rewritten_count,
                }
                for rule in rewrite_results
            ],
            "model_path": str(candidate_model),
            "vela_model_path": str(generated_vela_path),
            "vela_info_path": str(candidate_info_path),
            "summary": candidate_summary,
            "score": _candidate_score(candidate_summary, generated_vela_path),
        }
        candidate_results.append(entry)
        if best is None or entry["score"] > best["score"]:
            best = entry

    optimized_summary = dict(best["summary"]) if best else None
    improved = best is not None and _is_improvement(baseline_summary, optimized_summary or {})

    if improved:
        final_vela_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best["vela_model_path"], final_vela_path)
        optimized_info_path = Path(best["vela_info_path"])
        optimized_vela_path: Path | None = final_vela_path
        notes = [
            f"npu optimizer selected {best['name']}: "
            f"{_format_npu(baseline_summary)} -> {_format_npu(optimized_summary or {})}"
        ]
    else:
        optimized_info_path = None
        optimized_vela_path = None
        notes = ["npu optimizer found no PC-side Vela improvement"]

    result = NpuOptimizationResult(
        applied=improved,
        triggered=True,
        reason=_trigger_reason(baseline_summary, config.npu_threshold_pct),
        best_candidate=str(best["name"]) if best else None,
        baseline_info_path=baseline_info,
        optimized_info_path=optimized_info_path,
        baseline_vela_path=baseline_vela,
        optimized_vela_path=optimized_vela_path,
        trace_path=trace_path,
        notes=notes,
        baseline_summary=baseline_summary,
        optimized_summary=optimized_summary,
    )
    _write_trace(trace_path, result, candidate_results)
    return result


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _find_generated_vela_model(candidate_model: Path, vela_dir: Path) -> Path:
    matches = sorted(vela_dir.glob(f"{candidate_model.stem}*_vela.tflite"))
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one optimized vela model for {candidate_model.stem}, got {len(matches)}"
        )
    return matches[0]


def _candidate_score(summary: dict[str, Any], vela_path: Path) -> tuple[float, int, int, int]:
    return (
        _float_value(summary.get("npu_utilization_pct"), default=0.0),
        -int(summary.get("cpu_ops") or 0),
        -len(summary.get("cpu_fallback_ops") or []),
        -vela_path.stat().st_size if vela_path.exists() else 0,
    )


def _is_improvement(baseline: dict[str, Any], optimized: dict[str, Any]) -> bool:
    baseline_npu = _float_value(baseline.get("npu_utilization_pct"), default=0.0)
    optimized_npu = _float_value(optimized.get("npu_utilization_pct"), default=0.0)
    baseline_cpu = int(baseline.get("cpu_ops") or 0)
    optimized_cpu = int(optimized.get("cpu_ops") or 0)
    baseline_fallback = set(baseline.get("cpu_fallback_ops") or [])
    optimized_fallback = set(optimized.get("cpu_fallback_ops") or [])
    return (
        optimized_npu > baseline_npu
        or optimized_cpu < baseline_cpu
        or len(optimized_fallback) < len(baseline_fallback)
    )


def _trigger_reason(summary: dict[str, Any], threshold_pct: float) -> str:
    npu_pct = _float_value(summary.get("npu_utilization_pct"), default=0.0)
    fallback_ops = list(summary.get("cpu_fallback_ops") or [])
    real_fallback_ops = [op for op in fallback_ops if op != "Passthrough"]
    reasons: list[str] = []
    if npu_pct < threshold_pct:
        reasons.append(f"npu_utilization_pct {npu_pct} < {threshold_pct}")
    if real_fallback_ops:
        reasons.append("non-passthrough fallback ops: " + ",".join(real_fallback_ops))
    return "; ".join(reasons) or "optimization requested"


def _write_trace(
    trace_path: Path,
    result: NpuOptimizationResult,
    candidate_results: list[dict[str, Any]],
) -> None:
    payload = {
        "triggered": result.triggered,
        "reason": result.reason,
        "best_candidate": result.best_candidate,
        "baseline": result.baseline_summary,
        "optimized": result.optimized_summary,
        "baseline_info_path": _path_or_none(result.baseline_info_path),
        "optimized_info_path": _path_or_none(result.optimized_info_path),
        "baseline_vela_path": _path_or_none(result.baseline_vela_path),
        "optimized_vela_path": _path_or_none(result.optimized_vela_path),
        "candidates": [_json_safe(candidate) for candidate in candidate_results],
        "board_verification": {
            "required": True,
            "status": "pending",
        },
    }
    trace_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items() if key != "score"}
    return value


def _float_value(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _format_npu(summary: dict[str, Any]) -> str:
    return f"{summary.get('npu_utilization_pct', '-')}% NPU, fallback={summary.get('cpu_fallback_ops', [])}"


def _path_or_none(path: Path | None) -> str | None:
    return str(path) if path is not None else None
