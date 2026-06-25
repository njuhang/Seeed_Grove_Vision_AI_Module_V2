import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _size_kb(path: str | Path) -> float:
    return round(Path(path).stat().st_size / 1024, 1)


def _plan_models(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    models: dict[str, dict[str, Any]] = {}
    for batch in plan.get("batches", []):
        for model in batch.get("models", []):
            models[model["name"]] = {
                **model,
                "batch_index": batch.get("batch_index"),
            }
    return models


def build_rows(
    plan: dict[str, Any],
    conversion_summary: dict[str, Any],
    vela_dir: str | Path,
) -> list[dict[str, Any]]:
    planned = _plan_models(plan)
    rows: list[dict[str, Any]] = []
    for result in conversion_summary.get("results", []):
        name = result["name"]
        plan_entry = planned.get(name, {})
        vela_info_path = Path(vela_dir) / f"{name}.vela_info.json"
        vela_info = _load_json(vela_info_path) if vela_info_path.exists() else {}
        int8_path = result.get("int8_model_path")
        vela_path = result.get("vela_model_path")
        rows.append(
            {
                "Model": name,
                "Task": result.get("task", "unknown"),
                "Source": result.get("source_type", "unknown"),
                "Status": result.get("status", "unknown"),
                "Batch": plan_entry.get("batch_index", "-"),
                "Flash": plan_entry.get("flash_address", "-"),
                "Int8 KB": _size_kb(int8_path) if int8_path else "-",
                "Vela KB": _size_kb(vela_path) if vela_path else "-",
                "NPU%": vela_info.get("npu_utilization_pct", "-"),
                "CPU Fallback": ",".join(vela_info.get("cpu_fallback_ops", [])) or "-",
            }
        )
    return rows


def render_markdown(rows: list[dict[str, Any]], title: str) -> str:
    header = (
        f"# {title}\n\n"
        "| Model | Task | Source | Status | Batch | Flash | Int8 KB | Vela KB | NPU% | CPU Fallback |\n"
        "|-------|------|--------|--------|-------|-------|---------|---------|------|--------------|"
    )
    body = "\n".join(
        "| {Model} | {Task} | {Source} | {Status} | {Batch} | {Flash} | {Int8 KB} | {Vela KB} | {NPU%} | {CPU Fallback} |".format_map(
            {key: str(value).replace("|", "\\|") for key, value in row.items()}
        )
        for row in rows
    )
    return f"{header}\n{body}\n"


def write_summary(
    plan_path: str | Path,
    conversion_summary_path: str | Path,
    vela_dir: str | Path,
    output_path: str | Path,
    title: str = "Phase 6 Tier-2 OD Conversion Summary",
) -> Path:
    rows = build_rows(
        _load_json(plan_path),
        _load_json(conversion_summary_path),
        vela_dir,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_markdown(rows, title), encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--conversion-summary", required=True)
    parser.add_argument("--vela-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="Phase 6 Tier-2 OD Conversion Summary")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(
        write_summary(
            plan_path=args.plan,
            conversion_summary_path=args.conversion_summary,
            vela_dir=args.vela_dir,
            output_path=args.output,
            title=args.title,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
