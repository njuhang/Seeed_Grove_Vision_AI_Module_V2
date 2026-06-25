import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.flash_runner.flash_firmware import flash_firmware_image
from tools.flash_runner.flash_models import flash_model_batch
from tools.flash_runner.model_table import build_model_table
from tools.flash_runner.serial_capture import capture_json_objects
from tools.model_converter.convert import convert_manifest
from tools.model_converter.model_registry import ModelSpec, load_models
from tools.model_converter.convert import write_summary as write_conversion_summary
from tools.report.generate_report import load_conversion_summary, load_payload, write_report


@dataclass(frozen=True)
class ResolvedModelArtifact:
    spec: ModelSpec
    model_path: Path
    model_size_bytes: int


RUNTIME_FLASH_BASE = 0x00400000
FLASH_ALIGN = 0x1000


def _align_up(value: int, alignment: int = FLASH_ALIGN) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def compute_batches(models: list[tuple[str, int, int]], flash_capacity: int) -> list[list[tuple[str, int, int]]]:
    batches: list[list[tuple[str, int, int]]] = []
    current: list[tuple[str, int, int]] = []
    current_usage = 0

    for item in models:
        if not current:
            current.append(item)
            current_usage = _align_up(item[2])
            continue

        item_usage = _align_up(item[2])
        if current_usage + item_usage > flash_capacity:
            batches.append(current)
            current = []
            current_usage = 0

        if not current:
            current_usage = item_usage
        else:
            current_usage += item_usage
        current.append(item)

    if current:
        batches.append(current)
    return batches


def resolve_model_artifacts(
    repo_root: str | Path,
    manifest_path: str | Path = "configs/models.yaml",
    *,
    artifact_root: str | Path = "artifacts",
    skip_missing: bool = False,
    tiers: set[int] | None = None,
) -> list[ResolvedModelArtifact]:
    repo_root = Path(repo_root)
    manifest_path = Path(manifest_path)
    if not manifest_path.is_absolute():
        manifest_path = repo_root / manifest_path
    artifact_root = Path(artifact_root)
    if not artifact_root.is_absolute():
        artifact_root = repo_root / artifact_root

    resolved: list[ResolvedModelArtifact] = []
    selected_tiers = tiers or {1}
    for spec in load_models(manifest_path):
        if spec.tier not in selected_tiers:
            continue
        try:
            model_path = _resolve_artifact_path(repo_root, spec, artifact_root=artifact_root)
        except ValueError:
            if skip_missing:
                continue
            raise
        resolved.append(
            ResolvedModelArtifact(
                spec=spec,
                model_path=model_path,
                model_size_bytes=model_path.stat().st_size,
            )
        )
    return resolved


def _resolve_artifact_path(repo_root: Path, spec: ModelSpec, *, artifact_root: Path) -> Path:
    artifact_dir = artifact_root / spec.name
    artifact_matches = sorted(artifact_dir.glob("*_vela.tflite"))
    if len(artifact_matches) == 1:
        return artifact_matches[0]
    if len(artifact_matches) > 1:
        raise ValueError(f"multiple candidate artifacts found for {spec.name} in {artifact_dir}")

    if spec.model_zoo_ref:
        matches = list((repo_root / "model_zoo").rglob(spec.model_zoo_ref))
        if len(matches) != 1:
            raise ValueError(
                f"expected exactly one model_zoo match for {spec.name}: {spec.model_zoo_ref}, got {len(matches)}"
            )
        return matches[0]
    raise ValueError(f"no artifact found for {spec.name}")


def _resolve_model_zoo_path(repo_root: Path, spec: ModelSpec) -> Path:
    if not spec.model_zoo_ref:
        raise ValueError(f"model_zoo_ref is required for {spec.name}")
    matches = list((repo_root / "model_zoo").rglob(spec.model_zoo_ref))
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one model_zoo match for {spec.name}: {spec.model_zoo_ref}, got {len(matches)}"
        )
    return matches[0]


def write_execution_plan(
    repo_root: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
    flash_capacity: int,
    conversion_summary_path: str | Path | None = None,
    include_model_zoo_baselines: bool = False,
    tiers: set[int] | None = None,
) -> Path:
    repo_root = Path(repo_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved = resolve_model_artifacts(
        repo_root,
        manifest_path,
        artifact_root=output_dir,
        skip_missing=True,
        tiers=tiers,
    )
    conversion_by_name: dict[str, dict] = {}
    if include_model_zoo_baselines and conversion_summary_path is not None:
        conversion_by_name = {
            result["name"]: result
            for result in load_conversion_summary(conversion_summary_path)
            if "name" in result
        }

    plan_entries: list[dict[str, object]] = []
    for item in resolved:
        conversion = conversion_by_name.get(item.spec.name, {})
        can_compare_against_model_zoo = (
            include_model_zoo_baselines
            and item.spec.model_zoo_ref is not None
            and conversion.get("status") == "exported_from_source"
        )
        if can_compare_against_model_zoo:
            plan_entries.append(
                {
                    "board_name": f"{item.spec.name} (ours)",
                    "spec": item.spec,
                    "model_path": item.model_path,
                    "model_size_bytes": item.model_size_bytes,
                }
            )
            model_zoo_path = _resolve_model_zoo_path(repo_root, item.spec)
            plan_entries.append(
                {
                    "board_name": f"{item.spec.name} (model_zoo)",
                    "spec": item.spec,
                    "model_path": model_zoo_path,
                    "model_size_bytes": model_zoo_path.stat().st_size,
                }
            )
            continue

        plan_entries.append(
            {
                "board_name": item.spec.name,
                "spec": item.spec,
                "model_path": item.model_path,
                "model_size_bytes": item.model_size_bytes,
            }
        )

    flat = [
        (str(entry["board_name"]), int(entry["spec"].flash_address), int(entry["model_size_bytes"]))
        for entry in plan_entries
    ]
    batch_slices = compute_batches(flat, flash_capacity)
    entry_by_name = {str(entry["board_name"]): entry for entry in plan_entries}

    batches_payload: list[dict] = []
    for batch_index, batch in enumerate(batch_slices):
        items = [entry_by_name[name] for name, _, _ in batch]
        table_path = output_dir / f"model_table_batch_{batch_index}.bin"
        packed_models: list[tuple[dict[str, object], int]] = []
        runtime_flash_addr = RUNTIME_FLASH_BASE
        for item in items:
            packed_models.append((item, runtime_flash_addr))
            runtime_flash_addr += _align_up(int(item["model_size_bytes"]))

        table_path.write_bytes(
            build_model_table(
                [
                    (
                        ModelSpec(
                            name=str(item["board_name"]),
                            task=item["spec"].task,
                            source=item["spec"].source,
                            input_shape=item["spec"].input_shape,
                            calibration_dataset=item["spec"].calibration_dataset,
                            tier=item["spec"].tier,
                            flash_address=flash_address,
                            model_zoo_ref=item["spec"].model_zoo_ref,
                            vela=item["spec"].vela,
                        ),
                        int(item["model_size_bytes"]),
                    )
                    for item, flash_address in packed_models
                ]
            )
        )
        batches_payload.append(
            {
                "batch_index": batch_index,
                "table_path": str(table_path),
                "models": [
                    {
                        "name": str(item["board_name"]),
                        "model_path": str(item["model_path"]),
                        "flash_address": f"0x{flash_address:x}",
                        "model_size_bytes": int(item["model_size_bytes"]),
                    }
                    for item, flash_address in packed_models
                ],
            }
        )

    plan = {
        "manifest": str(manifest_path),
        "tiers": sorted(tiers or {1}),
        "flash_capacity": flash_capacity,
        "batch_count": len(batches_payload),
        "batches": batches_payload,
    }
    plan_path = output_dir / "benchmark_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return plan_path


def prepare_artifacts(
    repo_root: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
    vela_dir: str | Path,
    skip_vela: bool = False,
    allow_model_zoo_ref: bool = True,
    isolate_source_exports: bool = False,
    tiers: set[int] | None = None,
) -> Path:
    results = convert_manifest(
        manifest_path=manifest_path,
        repo_root=repo_root,
        output_root=output_dir,
        vela_dir=vela_dir,
        skip_vela=skip_vela,
        allow_model_zoo_ref=allow_model_zoo_ref,
        isolate_source_exports=isolate_source_exports,
        tiers=tiers,
    )
    return write_conversion_summary(results, output_dir)


def capture_board_result(
    *,
    port: str,
    output_dir: str | Path,
    flash_baudrate: int = 921600,
    serial_timeout: float = 60.0,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payloads = capture_json_objects(
        port,
        baudrate=flash_baudrate,
        timeout=serial_timeout,
        log_path=output_dir / "serial_benchmark_capture.log",
    )
    latest_payload = payloads[-1]
    board_result_path = output_dir / "benchmark_result_latest.json"
    board_result_path.write_text(json.dumps(latest_payload, indent=2), encoding="utf-8")
    return board_result_path


def run_execution_plan(
    plan_path: str | Path,
    *,
    port: str,
    output_dir: str | Path,
    flash_baudrate: int = 921600,
    serial_timeout: float = 60.0,
    auto_reset: bool = False,
    flash_model_chunk_size: int | None = None,
    python_executable: str = "python",
    xmodem_script: str | Path = "xmodem/xmodem_send.py",
) -> Path:
    plan_path = Path(plan_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    benchmark_meta: dict | None = None
    models_payload: list[dict] = []

    for batch in plan.get("batches", []):
        flash_model_batch(
            port=port,
            table_path=batch["table_path"],
            models=[
                (model["model_path"], int(model["flash_address"], 0))
                for model in batch["models"]
            ],
            baudrate=flash_baudrate,
            auto_reset=auto_reset,
            model_chunk_size=flash_model_chunk_size,
            python_executable=python_executable,
            xmodem_script=xmodem_script,
        )
        payloads = capture_json_objects(
            port,
            baudrate=flash_baudrate,
            timeout=serial_timeout,
            log_path=output_dir / f"serial_benchmark_capture_batch{batch['batch_index']}.log",
        )
        latest_payload = payloads[-1]
        if benchmark_meta is None:
            benchmark_meta = latest_payload.get("benchmark", {})
        models_payload.extend(latest_payload.get("models", []))

    merged_payload = {
        "benchmark": benchmark_meta or {},
        "models": models_payload,
    }
    board_result_path = output_dir / "benchmark_result_latest.json"
    board_result_path.write_text(json.dumps(merged_payload, indent=2), encoding="utf-8")
    return board_result_path


def run_board_benchmark(
    plan_path: str | Path,
    *,
    port: str,
    output_dir: str | Path,
    flash_firmware_first: bool = False,
    skip_flash: bool = False,
    flash_baudrate: int = 921600,
    serial_timeout: float = 60.0,
    auto_reset: bool = False,
    flash_model_chunk_size: int | None = None,
    python_executable: str = "python",
    xmodem_script: str | Path = "xmodem/xmodem_send.py",
) -> Path:
    if flash_firmware_first:
        flash_firmware_image(
            port=port,
            baudrate=flash_baudrate,
            auto_reset=auto_reset,
            python_executable=python_executable,
            xmodem_script=xmodem_script,
        )
    if skip_flash:
        return capture_board_result(
            port=port,
            output_dir=output_dir,
            flash_baudrate=flash_baudrate,
            serial_timeout=serial_timeout,
        )
    return run_execution_plan(
        plan_path,
        port=port,
        output_dir=output_dir,
        flash_baudrate=flash_baudrate,
        serial_timeout=serial_timeout,
        auto_reset=auto_reset,
        flash_model_chunk_size=flash_model_chunk_size,
        python_executable=python_executable,
        xmodem_script=xmodem_script,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="configs/models.yaml")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--flash-capacity", default="0x400000")
    parser.add_argument(
        "--tiers",
        default="1",
        help="comma-separated manifest tiers to include in artifact planning, e.g. 1,2 or 2",
    )
    parser.add_argument("--run-convert", action="store_true")
    parser.add_argument("--skip-convert-vela", action="store_true")
    parser.add_argument("--no-model-zoo-ref", action="store_true")
    parser.add_argument("--isolate-source-exports", action="store_true")
    parser.add_argument("--board-result", dest="board_result")
    parser.add_argument("--from-board-result", dest="board_result")
    parser.add_argument("--port")
    parser.add_argument("--flash-firmware-first", action="store_true")
    parser.add_argument("--skip-flash", action="store_true")
    parser.add_argument("--flash-baudrate", default="921600")
    parser.add_argument("--serial-timeout", type=float, default=60.0)
    parser.add_argument("--auto-reset", action="store_true")
    parser.add_argument("--flash-model-chunk-size")
    parser.add_argument("--python-executable", default="python")
    parser.add_argument("--xmodem-script", default="xmodem/xmodem_send.py")
    parser.add_argument("--vela-dir")
    parser.add_argument("--report-prefix", default="artifacts/benchmark_report")
    return parser.parse_args()


def parse_tiers(raw: str) -> set[int]:
    tiers = {int(item.strip()) for item in raw.split(",") if item.strip()}
    if not tiers:
        raise ValueError("--tiers must include at least one integer tier")
    return tiers


def main() -> int:
    args = parse_args()
    repo_root = Path.cwd()
    flash_capacity = int(args.flash_capacity, 0)
    tiers = parse_tiers(args.tiers)
    vela_dir = args.vela_dir or str(Path(args.output_dir) / "vela")
    conversion_summary_path: Path | None = None

    if args.run_convert:
        conversion_summary_path = prepare_artifacts(
            repo_root=repo_root,
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            vela_dir=vela_dir,
            skip_vela=args.skip_convert_vela,
            allow_model_zoo_ref=not args.no_model_zoo_ref,
            isolate_source_exports=args.isolate_source_exports,
            tiers=tiers,
        )
        print(conversion_summary_path)

    if conversion_summary_path is None:
        discovered_summary = Path(args.output_dir) / "conversion_summary.json"
        if discovered_summary.exists():
            conversion_summary_path = discovered_summary

    plan_path = write_execution_plan(
        repo_root=repo_root,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        flash_capacity=flash_capacity,
        conversion_summary_path=conversion_summary_path,
        include_model_zoo_baselines=True,
        tiers=tiers,
    )
    print(plan_path)

    board_result_path = args.board_result
    if args.port:
        board_result_path = str(
            run_board_benchmark(
                plan_path,
                port=args.port,
                output_dir=args.output_dir,
                flash_firmware_first=args.flash_firmware_first,
                skip_flash=args.skip_flash,
                flash_baudrate=int(args.flash_baudrate, 0),
                serial_timeout=args.serial_timeout,
                auto_reset=args.auto_reset,
                flash_model_chunk_size=int(args.flash_model_chunk_size, 0) if args.flash_model_chunk_size else None,
                python_executable=args.python_executable,
                xmodem_script=args.xmodem_script,
            )
        )
        print(board_result_path)

    if board_result_path:
        payload = load_payload(board_result_path)
        markdown_path, csv_path = write_report(
            payload,
            manifest_path=args.manifest,
            output_prefix=args.report_prefix,
            vela_dir=vela_dir,
        )
        print(markdown_path)
        print(csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
