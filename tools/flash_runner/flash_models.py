import argparse
import json
import subprocess
from pathlib import Path


def _cli_path(path: str | Path) -> str:
    raw = Path(path).as_posix() if isinstance(path, Path) else path.replace("\\", "/")
    if raw.startswith("/mnt/") and len(raw) > 6 and raw[5].isalpha() and raw[6] == "/":
        return f"{raw[5].upper()}:{raw[6:]}"
    return raw


def _append_model_arg(args: list[str], model_path: str | Path, flash_address: int, model_offset: int = 0) -> None:
    args.extend(
        [
            "--model",
            f"{_cli_path(model_path)} 0x{flash_address:x} 0x{model_offset:05x}",
        ]
    )


def build_model_args(
    models: list[tuple[str, int]],
    *,
    model_chunk_size: int | None = None,
    chunk_root: str | Path | None = None,
) -> list[str]:
    args: list[str] = []
    for model_path, flash_address in models:
        path = Path(model_path)
        if (
            model_chunk_size is None
            or model_chunk_size <= 0
            or not path.exists()
            or path.stat().st_size <= model_chunk_size
        ):
            _append_model_arg(args, model_path, flash_address)
            continue

        if chunk_root is None:
            raise ValueError("chunk_root is required when model_chunk_size is enabled")

        chunk_dir = Path(chunk_root) / path.stem
        chunk_dir.mkdir(parents=True, exist_ok=True)
        with path.open("rb") as handle:
            chunk_index = 0
            model_offset = 0
            while True:
                payload = handle.read(model_chunk_size)
                if not payload:
                    break
                chunk_path = chunk_dir / f"{path.stem}_chunk_{chunk_index:02d}.bin"
                chunk_path.write_bytes(payload)
                _append_model_arg(args, chunk_path, flash_address, model_offset)
                chunk_index += 1
                model_offset += len(payload)
    return args


def model_table_path(batch_index: int) -> Path:
    return Path("artifacts") / f"model_table_batch_{batch_index}.bin"


def build_flash_command(
    port: str,
    table_path: str | Path,
    models: list[tuple[str, int]],
    *,
    baudrate: int = 921600,
    protocol: str = "xmodem",
    auto_reset: bool = False,
    python_executable: str = "python",
    xmodem_script: str | Path = "xmodem/xmodem_send.py",
    model_chunk_size: int | None = None,
    chunk_root: str | Path | None = None,
) -> list[str]:
    command = [
        python_executable,
        str(xmodem_script),
        "--port",
        port,
        "--baudrate",
        str(baudrate),
        "--protocol",
        protocol,
    ]
    if auto_reset:
        command.append("--auto-reset")
    _append_model_arg(command, str(table_path), 0x00200000)
    command.extend(build_model_args(models, model_chunk_size=model_chunk_size, chunk_root=chunk_root))
    return command


def flash_model_batch(
    port: str,
    table_path: str | Path,
    models: list[tuple[str, int]],
    *,
    baudrate: int = 921600,
    protocol: str = "xmodem",
    auto_reset: bool = False,
    python_executable: str = "python",
    xmodem_script: str | Path = "xmodem/xmodem_send.py",
    model_chunk_size: int | None = None,
):
    chunk_root = None
    if model_chunk_size is not None and model_chunk_size > 0:
        chunk_root = Path(table_path).parent / "model_chunks"
    command = build_flash_command(
        port=port,
        table_path=table_path,
        models=models,
        baudrate=baudrate,
        protocol=protocol,
        auto_reset=auto_reset,
        python_executable=python_executable,
        xmodem_script=xmodem_script,
        model_chunk_size=model_chunk_size,
        chunk_root=chunk_root,
    )
    return subprocess.run(command, check=True)


def flash_batch_from_plan(
    plan_path: str | Path,
    *,
    batch_index: int,
    port: str,
    baudrate: int = 921600,
    protocol: str = "xmodem",
    auto_reset: bool = False,
    python_executable: str = "python",
    xmodem_script: str | Path = "xmodem/xmodem_send.py",
    model_chunk_size: int | None = None,
):
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    for batch in plan.get("batches", []):
        if int(batch["batch_index"]) != batch_index:
            continue
        return flash_model_batch(
            port=port,
            table_path=batch["table_path"],
            models=[
                (model["model_path"], int(model["flash_address"], 0))
                for model in batch["models"]
            ],
            baudrate=baudrate,
            protocol=protocol,
            auto_reset=auto_reset,
            python_executable=python_executable,
            xmodem_script=xmodem_script,
            model_chunk_size=model_chunk_size,
        )
    raise ValueError(f"batch_index {batch_index} not found in {plan_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    parser.add_argument("--plan", default="artifacts/benchmark_plan.json")
    parser.add_argument("--batch", required=True, type=int)
    parser.add_argument("--baudrate", default="921600")
    parser.add_argument("--protocol", default="xmodem")
    parser.add_argument("--auto-reset", action="store_true")
    parser.add_argument("--python-executable", default="python")
    parser.add_argument("--xmodem-script", default="xmodem/xmodem_send.py")
    parser.add_argument("--model-chunk-size")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    flash_batch_from_plan(
        args.plan,
        batch_index=args.batch,
        port=args.port,
        baudrate=int(args.baudrate, 0),
        protocol=args.protocol,
        auto_reset=args.auto_reset,
        python_executable=args.python_executable,
        xmodem_script=args.xmodem_script,
        model_chunk_size=int(args.model_chunk_size, 0) if args.model_chunk_size else None,
    )
    print(args.plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
