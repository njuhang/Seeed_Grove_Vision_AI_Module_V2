import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.model_converter.model_registry import load_models

# Conversion statuses that produced a runnable on-device artifact. Anything else
# in the conversion summary is treated as a failure and surfaced in the report.
SUCCESS_CONVERSION_STATUSES = {"staged_model_zoo_ref", "converted", "exported_from_source"}

OURS_SUFFIX = " (ours)"
MODEL_ZOO_SUFFIX = " (model_zoo)"
DIFF_SUFFIX = " (diff)"
SIZE_TREND_SUFFIX = " (size trend)"
SIZE_ORDER = {"n": 0, "s": 1, "m": 2, "l": 3, "x": 4}
SIZE_VARIANT_RE = re.compile(r"^(?P<family>yolo(?:v\d+|\d+))(?P<size>[nsmxl])_(?P<rest>.+)$")


def load_payload(input_path: str | Path) -> dict[str, Any]:
    return json.loads(Path(input_path).read_text(encoding="utf-8-sig"))


def load_conversion_summary(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    return list(payload.get("results", []))


def load_manifest_tasks(manifest_path: str | Path) -> dict[str, str]:
    return {model.name: model.task for model in load_models(manifest_path)}


def load_vela_infos(vela_dir: str | Path) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for path in Path(vela_dir).glob("*.vela_info.json"):
        model_name = path.name.removesuffix(".vela_info.json")
        payloads[model_name] = json.loads(path.read_text(encoding="utf-8-sig"))
    return payloads


def _split_comparison_name(name: str) -> tuple[str, str | None]:
    if name.endswith(OURS_SUFFIX):
        return name[: -len(OURS_SUFFIX)], "ours"
    if name.endswith(MODEL_ZOO_SUFFIX):
        return name[: -len(MODEL_ZOO_SUFFIX)], "model_zoo"
    if name.endswith(DIFF_SUFFIX):
        return name[: -len(DIFF_SUFFIX)], "diff"
    return name, None


def _lookup_vela_info(
    board_name: str,
    vela_infos: dict[str, dict[str, Any]],
    conversion_by_name: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    base_name, variant = _split_comparison_name(board_name)
    conversion = conversion_by_name.get(base_name, {})
    optimization = conversion.get("npu_optimization") or {}
    if variant in {None, "ours"} and optimization.get("applied"):
        optimized_summary = optimization.get("optimized_summary") or {}
        if optimized_summary:
            return optimized_summary
    if variant == "ours":
        return vela_infos.get(base_name, {})
    if variant == "model_zoo":
        model_zoo_ref = conversion.get("model_zoo_ref")
        if model_zoo_ref:
            vela_info = vela_infos.get(Path(model_zoo_ref).stem, {})
            input_model = str(vela_info.get("input_model", ""))
            if (
                input_model.endswith("_vela.tflite")
                and vela_info.get("npu_ops") == 0
                and vela_info.get("cpu_ops") == 1
                and vela_info.get("cpu_fallback_ops") == ["Passthrough"]
            ):
                return {}
            return vela_info
        return {}
    return vela_infos.get(board_name, vela_infos.get(base_name, {}))


def _delta_value(lhs: Any, rhs: Any) -> Any:
    if isinstance(lhs, (int, float)) and isinstance(rhs, (int, float)):
        return round(lhs - rhs, 1)
    return "-"


def _build_comparison_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ours_by_base: dict[str, dict[str, Any]] = {}
    model_zoo_by_base: dict[str, dict[str, Any]] = {}

    for row in rows:
        base_name, variant = _split_comparison_name(row["Model"])
        if variant == "ours":
            ours_by_base[base_name] = row
        elif variant == "model_zoo":
            model_zoo_by_base[base_name] = row

    comparison_rows: list[dict[str, Any]] = []
    for base_name in sorted(ours_by_base.keys() & model_zoo_by_base.keys()):
        ours = ours_by_base[base_name]
        model_zoo = model_zoo_by_base[base_name]
        ours_fallback = ours["CPU Fallback"]
        model_zoo_fallback = model_zoo["CPU Fallback"]
        if ours_fallback == model_zoo_fallback:
            fallback_diff = "same"
        else:
            fallback_diff = f"ours:{ours_fallback} | model_zoo:{model_zoo_fallback}"
        comparison_rows.append(
            {
                "Model": f"{base_name}{DIFF_SUFFIX}",
                "Task": ours["Task"],
                "Size (KB)": _delta_value(ours["Size (KB)"], model_zoo["Size (KB)"]),
                "Arena (KB)": _delta_value(ours["Arena (KB)"], model_zoo["Arena (KB)"]),
                "Avg (ms)": _delta_value(ours["Avg (ms)"], model_zoo["Avg (ms)"]),
                "Min (ms)": _delta_value(ours["Min (ms)"], model_zoo["Min (ms)"]),
                "Max (ms)": _delta_value(ours["Max (ms)"], model_zoo["Max (ms)"]),
                "NPU%": _delta_value(ours["NPU%"], model_zoo["NPU%"]),
                "CPU Fallback": fallback_diff,
                "NPU Optimization": ours.get("NPU Optimization", "-"),
                "Status": "comparison",
                "Reason": "ours - model_zoo",
            }
        )
    return comparison_rows


def _format_trend_value(entries: list[tuple[str, dict[str, Any]]], field: str) -> str:
    return ", ".join(f"{size}:{row[field]}" for size, row in entries)


def _build_size_trend_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[tuple[str, dict[str, Any]]]] = {}
    for row in rows:
        base_name, variant = _split_comparison_name(row["Model"])
        if variant in {"model_zoo", "diff"} or row["Status"] != "ok":
            continue
        match = SIZE_VARIANT_RE.match(base_name)
        if not match:
            continue
        key = (match.group("family"), match.group("rest"), row["Task"])
        grouped.setdefault(key, []).append((match.group("size"), row))

    trend_rows: list[dict[str, Any]] = []
    for (family, rest, task), entries in sorted(grouped.items()):
        unique_by_size = {size: row for size, row in entries}
        if len(unique_by_size) < 2:
            continue
        ordered = sorted(unique_by_size.items(), key=lambda item: SIZE_ORDER[item[0]])
        trend_rows.append(
            {
                "Model": f"{family}_{rest}{SIZE_TREND_SUFFIX}",
                "Task": task,
                "Size (KB)": _format_trend_value(ordered, "Size (KB)"),
                "Arena (KB)": _format_trend_value(ordered, "Arena (KB)"),
                "Avg (ms)": _format_trend_value(ordered, "Avg (ms)"),
                "Min (ms)": _format_trend_value(ordered, "Min (ms)"),
                "Max (ms)": _format_trend_value(ordered, "Max (ms)"),
                "NPU%": _format_trend_value(ordered, "NPU%"),
                "CPU Fallback": _format_trend_value(ordered, "CPU Fallback"),
                "NPU Optimization": _format_trend_value(ordered, "NPU Optimization"),
                "Status": "trend",
                "Reason": "ordered by model size variant",
            }
        )
    return trend_rows


def merge_result_row(
    board_result: dict[str, Any],
    vela_info: dict[str, Any],
    manifest_task: str | None = None,
    conversion_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    task = board_result.get("task")
    if not task or task == "unknown":
        task = manifest_task or "unknown"

    fallback_ops = vela_info.get("cpu_fallback_ops", [])
    optimization = (conversion_result or {}).get("npu_optimization") or {}
    status = board_result["status"]
    reason = board_result.get("status_reason") or "-"
    if optimization.get("applied") and status != "ok":
        status = "npu_opt_board_failed"
        baseline_path = optimization.get("baseline_vela_path") or "-"
        reason = f"{reason}; baseline fallback: {baseline_path}"
    return {
        "Model": board_result["name"],
        "Task": task,
        "Size (KB)": round(board_result["model_size_bytes"] / 1024, 1),
        "Arena (KB)": round(board_result["arena_used_bytes"] / 1024, 1) if "arena_used_bytes" in board_result else "-",
        "Avg (ms)": board_result.get("latency_ms", {}).get("avg", "-"),
        "Min (ms)": board_result.get("latency_ms", {}).get("min", "-"),
        "Max (ms)": board_result.get("latency_ms", {}).get("max", "-"),
        "NPU%": vela_info.get("npu_utilization_pct", "-"),
        "CPU Fallback": ",".join(fallback_ops) or "-",
        "NPU Optimization": _format_npu_optimization(optimization),
        "Status": status,
        "Reason": reason,
    }


def merge_failure_row(
    conversion_result: dict[str, Any],
    manifest_task: str | None = None,
) -> dict[str, Any]:
    task = conversion_result.get("task") or manifest_task or "unknown"
    notes = conversion_result.get("notes") or []
    missing = conversion_result.get("missing_dependencies") or []

    reasons: list[str] = list(notes)
    if missing:
        reasons.append("missing: " + ", ".join(missing))
    reason = "; ".join(reasons) or conversion_result.get("status", "failed")

    return {
        "Model": conversion_result["name"],
        "Task": task,
        "Size (KB)": "-",
        "Arena (KB)": "-",
        "Avg (ms)": "-",
        "Min (ms)": "-",
        "Max (ms)": "-",
        "NPU%": "-",
        "CPU Fallback": "-",
        "NPU Optimization": _format_npu_optimization(conversion_result.get("npu_optimization") or {}),
        "Status": conversion_result.get("status", "failed"),
        "Reason": reason,
    }


def _format_npu_optimization(optimization: dict[str, Any]) -> str:
    if not optimization:
        return "-"
    if not optimization.get("triggered") and not optimization.get("applied"):
        return "-"
    if not optimization.get("applied"):
        return optimization.get("reason") or "no PC-side improvement"

    baseline = optimization.get("baseline_summary") or {}
    optimized = optimization.get("optimized_summary") or {}
    baseline_npu = baseline.get("npu_utilization_pct", "-")
    optimized_npu = optimized.get("npu_utilization_pct", "-")
    baseline_fallback = ",".join(baseline.get("cpu_fallback_ops") or []) or "-"
    optimized_fallback = ",".join(optimized.get("cpu_fallback_ops") or []) or "-"
    candidate = optimization.get("best_candidate") or "optimized"
    return f"{candidate}: {baseline_npu}->{optimized_npu}%; fallback {baseline_fallback}->{optimized_fallback}"


def build_rows(
    payload: dict[str, Any],
    manifest_tasks: dict[str, str],
    vela_infos: dict[str, dict[str, Any]],
    conversion_results: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    board_names: set[str] = set()
    conversion_by_name = {
        result["name"]: result
        for result in conversion_results or []
        if "name" in result
    }
    for model_result in payload.get("models", []):
        base_name, _ = _split_comparison_name(model_result["name"])
        rows.append(
            merge_result_row(
                model_result,
                _lookup_vela_info(model_result["name"], vela_infos, conversion_by_name),
                manifest_task=manifest_tasks.get(model_result["name"], manifest_tasks.get(base_name)),
                conversion_result=conversion_by_name.get(base_name),
            )
        )
        board_names.add(model_result["name"])

    for conversion in conversion_results or []:
        name = conversion.get("name")
        if name is None or name in board_names:
            continue
        if conversion.get("status") in SUCCESS_CONVERSION_STATUSES:
            continue
        rows.append(
            merge_failure_row(
                conversion,
                manifest_task=manifest_tasks.get(name),
            )
        )
    return rows + _build_comparison_rows(rows) + _build_size_trend_rows(rows)


def payload_rows(
    payload: dict[str, Any],
    manifest_tasks: dict[str, str],
    vela_infos: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    return build_rows(payload, manifest_tasks, vela_infos)


def render_markdown(rows: list[dict[str, Any]], title: str) -> str:
    def _escape_cell(value: Any) -> str:
        return str(value).replace("|", "\\|")

    header = (
        f"# {title}\n\n"
        "| Model | Task | Size (KB) | Arena (KB) | Avg (ms) | Min (ms) | Max (ms) | NPU% | CPU Fallback | NPU Optimization | Status | Reason |\n"
        "|-------|------|-----------|------------|----------|----------|----------|------|--------------|------------------|--------|--------|"
    )
    body = "\n".join(
        "| {Model} | {Task} | {Size (KB)} | {Arena (KB)} | {Avg (ms)} | {Min (ms)} | {Max (ms)} | {NPU%} | {CPU Fallback} | {NPU Optimization} | {Status} | {Reason} |".format_map(
            {
                key: _escape_cell(value)
                for key, value in {"NPU Optimization": "-", **row}.items()
            }
        )
        for row in rows
    )
    return f"{header}\n{body}\n"


def write_report(
    payload: dict[str, Any],
    manifest_path: str | Path,
    output_prefix: str | Path,
    vela_dir: str | Path | None = None,
    conversion_summary_path: str | Path | None = None,
    title: str = "Himax HX6538 Model Benchmark Report",
) -> tuple[Path, Path]:
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    manifest_tasks = load_manifest_tasks(manifest_path)
    vela_infos = load_vela_infos(vela_dir) if vela_dir else {}

    conversion_results: list[dict[str, Any]] | None = None
    if conversion_summary_path is not None:
        conversion_results = load_conversion_summary(conversion_summary_path)
    else:
        discovered = output_prefix.parent / "conversion_summary.json"
        if discovered.exists():
            conversion_results = load_conversion_summary(discovered)

    rows = build_rows(payload, manifest_tasks, vela_infos, conversion_results)

    markdown_path = output_prefix.with_suffix(".md")
    csv_path = output_prefix.with_suffix(".csv")

    markdown_path.write_text(render_markdown(rows, title), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    return markdown_path, csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Benchmark JSON payload path")
    parser.add_argument("--manifest", default="configs/models.yaml")
    parser.add_argument("--vela-dir")
    parser.add_argument(
        "--conversion-summary",
        help="conversion_summary.json path; failed models are added to the report",
    )
    parser.add_argument("--output-prefix", default="artifacts/benchmark_report")
    parser.add_argument("--title", default="Himax HX6538 Model Benchmark Report")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = load_payload(args.input)
    markdown_path, csv_path = write_report(
        payload,
        manifest_path=args.manifest,
        output_prefix=args.output_prefix,
        vela_dir=args.vela_dir,
        conversion_summary_path=args.conversion_summary,
        title=args.title,
    )
    print(markdown_path)
    print(csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
