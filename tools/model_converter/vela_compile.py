import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HIMAX_VELA_CONFIG = REPO_ROOT / "configs" / "himax_vela.ini"
DEFAULT_HIMAX_VELA_SYSTEM_CONFIG = "My_Sys_Cfg"
DEFAULT_HIMAX_VELA_MEMORY_MODE = "My_Mem_Mode_Parent"

CPU_OPERATIONS_RE = re.compile(
    r"CPU (?:operations|operators)\s*(?:[=:]\s*|\s{2,})(\d+)",
    re.IGNORECASE,
)
NPU_OPERATIONS_RE = re.compile(
    r"NPU (?:operations|operators)\s*(?:[=:]\s*|\s{2,})(\d+)",
    re.IGNORECASE,
)
CPU_BREAKDOWN_RE = re.compile(r"^\s+([A-Z_]+):\s+\d+$", flags=re.MULTILINE)
CPU_OPERATOR_LINE_RE = re.compile(r"^\s*CPU:\s*([^=]+?)\s*=", flags=re.MULTILINE)


def default_vela_executable() -> str:
    explicit = os.environ.get("VELA_EXECUTABLE")
    if explicit:
        return explicit

    candidates = [
        Path.home() / "miniforge3" / "envs" / "vela" / "bin" / "vela",
        Path.home() / ".local" / "bin" / "vela",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return "vela"


def default_accelerator_config() -> str:
    return os.environ.get("VELA_ACCELERATOR_CONFIG", "ethos-u55-64")


def _normalize_optional_env(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    return stripped


def default_vela_config_path() -> str | None:
    explicit = _normalize_optional_env("VELA_CONFIG")
    if explicit is not None:
        return explicit
    if DEFAULT_HIMAX_VELA_CONFIG.exists():
        return str(DEFAULT_HIMAX_VELA_CONFIG)
    return None


def default_vela_system_config() -> str | None:
    explicit = _normalize_optional_env("VELA_SYSTEM_CONFIG")
    if explicit is not None:
        return explicit
    if _normalize_optional_env("VELA_CONFIG") is not None:
        return None
    if DEFAULT_HIMAX_VELA_CONFIG.exists():
        return DEFAULT_HIMAX_VELA_SYSTEM_CONFIG
    return None


def default_vela_memory_mode() -> str | None:
    explicit = _normalize_optional_env("VELA_MEMORY_MODE")
    if explicit is not None:
        return explicit
    if _normalize_optional_env("VELA_CONFIG") is not None:
        return None
    if DEFAULT_HIMAX_VELA_CONFIG.exists():
        return DEFAULT_HIMAX_VELA_MEMORY_MODE
    return None


def parse_vela_output(stdout: str) -> dict[str, Any]:
    cpu_match = CPU_OPERATIONS_RE.search(stdout)
    npu_match = NPU_OPERATIONS_RE.search(stdout)
    if cpu_match is None or npu_match is None:
        raise ValueError("unable to find CPU/NPU operator counts in vela output")

    cpu_ops = int(cpu_match.group(1))
    npu_ops = int(npu_match.group(1))

    fallback_matches = CPU_BREAKDOWN_RE.findall(stdout)
    if not fallback_matches:
        fallback_matches = [match.strip() for match in CPU_OPERATOR_LINE_RE.findall(stdout)]

    seen: set[str] = set()
    cpu_fallback_ops: list[str] = []
    for name in fallback_matches:
        if name not in seen:
            seen.add(name)
            cpu_fallback_ops.append(name)

    total_ops = cpu_ops + npu_ops
    utilization = 0.0 if total_ops == 0 else round((npu_ops / total_ops) * 100, 1)
    return {
        "cpu_ops": cpu_ops,
        "npu_ops": npu_ops,
        "npu_utilization_pct": utilization,
        "cpu_fallback_ops": cpu_fallback_ops,
    }


def _running_in_wsl() -> bool:
    return bool(os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"))


def windows_path_to_wsl(path: str | Path) -> str:
    resolved = str(Path(path).resolve())
    if _running_in_wsl():
        return resolved
    result = subprocess.run(
        ["wsl", "bash", "-lc", f"wslpath -a {shlex.quote(resolved)}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _normalize_command_arg(arg: str) -> str:
    candidate = Path(arg)
    if candidate.is_absolute() and candidate.exists():
        return windows_path_to_wsl(candidate)

    repo_candidate = (REPO_ROOT / candidate).resolve()
    if not candidate.is_absolute() and repo_candidate.exists():
        return windows_path_to_wsl(repo_candidate)
    return arg


def build_vela_command(
    model_path: str | Path,
    output_dir: str | Path,
    vela_executable: str | None = None,
    vela_executable_args: list[str] | tuple[str, ...] | None = None,
    accelerator_config: str | None = None,
    config_path: str | Path | None = None,
    system_config: str | None = None,
    memory_mode: str | None = None,
    optimise: str | None = None,
    tensor_allocator: str | None = None,
    extra_args: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    vela_executable = vela_executable or default_vela_executable()
    accelerator_config = accelerator_config or default_accelerator_config()
    config_path = config_path or default_vela_config_path()
    system_config = system_config or default_vela_system_config()
    memory_mode = memory_mode or default_vela_memory_mode()
    command = [
        vela_executable,
        *[_normalize_command_arg(arg) for arg in (vela_executable_args or [])],
        windows_path_to_wsl(model_path),
        "--output-dir",
        windows_path_to_wsl(output_dir),
        "--accelerator-config",
        accelerator_config,
        "--show-cpu-operations",
    ]
    if config_path is not None:
        command.extend(["--config", windows_path_to_wsl(config_path)])
    if system_config is not None:
        command.extend(["--system-config", system_config])
    if memory_mode is not None:
        command.extend(["--memory-mode", memory_mode])
    if optimise is not None:
        command.extend(["--optimise", optimise])
    if tensor_allocator is not None:
        command.extend(["--tensor-allocator", tensor_allocator])
    if extra_args:
        command.extend(_normalize_command_arg(arg) for arg in extra_args)
    if _running_in_wsl():
        return command
    return ["wsl", *command]


def compile_with_vela(
    model_path: str | Path,
    output_dir: str | Path,
    vela_executable: str | None = None,
    vela_executable_args: list[str] | tuple[str, ...] | None = None,
    accelerator_config: str | None = None,
    config_path: str | Path | None = None,
    system_config: str | None = None,
    memory_mode: str | None = None,
    optimise: str | None = None,
    tensor_allocator: str | None = None,
    extra_args: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    accelerator_config = accelerator_config or default_accelerator_config()
    config_path = config_path or default_vela_config_path()
    system_config = system_config or default_vela_system_config()
    memory_mode = memory_mode or default_vela_memory_mode()

    command = build_vela_command(
        model_path=model_path,
        output_dir=output_dir,
        vela_executable=vela_executable,
        vela_executable_args=vela_executable_args,
        accelerator_config=accelerator_config,
        config_path=config_path,
        system_config=system_config,
        memory_mode=memory_mode,
        optimise=optimise,
        tensor_allocator=tensor_allocator,
        extra_args=extra_args,
    )
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        details = "\n".join(part for part in [exc.output, exc.stderr] if part)
        if details:
            raise RuntimeError(f"vela compilation failed:\n{details}") from exc
        raise RuntimeError(f"vela compilation failed with exit code {exc.returncode}") from exc
    summary = parse_vela_output(result.stdout + result.stderr)
    summary["accelerator_config"] = accelerator_config
    summary["input_model"] = str(Path(model_path).resolve())
    summary["output_dir"] = str(output_dir.resolve())
    if config_path is not None:
        summary["config_path"] = str(Path(config_path).resolve())
    if system_config is not None:
        summary["system_config"] = system_config
    if memory_mode is not None:
        summary["memory_mode"] = memory_mode
    if optimise is not None:
        summary["optimise"] = optimise
    if tensor_allocator is not None:
        summary["tensor_allocator"] = tensor_allocator
    if vela_executable is not None:
        summary["vela_executable"] = vela_executable
    if vela_executable_args:
        summary["vela_executable_args"] = list(vela_executable_args)
    if extra_args:
        summary["extra_args"] = list(extra_args)
    return summary


def write_vela_artifacts(
    model_name: str,
    model_path: str | Path,
    output_dir: str | Path,
    vela_executable: str | None = None,
    vela_executable_args: list[str] | tuple[str, ...] | None = None,
    accelerator_config: str | None = None,
    config_path: str | Path | None = None,
    system_config: str | None = None,
    memory_mode: str | None = None,
    optimise: str | None = None,
    tensor_allocator: str | None = None,
    extra_args: list[str] | tuple[str, ...] | None = None,
) -> Path:
    summary = compile_with_vela(
        model_path=model_path,
        output_dir=output_dir,
        vela_executable=vela_executable,
        vela_executable_args=vela_executable_args,
        accelerator_config=accelerator_config,
        config_path=config_path,
        system_config=system_config,
        memory_mode=memory_mode,
        optimise=optimise,
        tensor_allocator=tensor_allocator,
        extra_args=extra_args,
    )
    out_path = Path(output_dir) / f"{model_name}.vela_info.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Input TFLite model path")
    parser.add_argument("--model-name", help="Logical model name for the sidecar file")
    parser.add_argument("--output-dir", default="artifacts/vela")
    parser.add_argument("--vela-executable", default=default_vela_executable())
    parser.add_argument("--accelerator-config", default=default_accelerator_config())
    parser.add_argument("--config", default=default_vela_config_path())
    parser.add_argument("--system-config", default=default_vela_system_config())
    parser.add_argument("--memory-mode", default=default_vela_memory_mode())
    parser.add_argument("--optimise")
    parser.add_argument("--tensor-allocator")
    parser.add_argument("--extra-arg", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_name = args.model_name or Path(args.model).stem
    out_path = write_vela_artifacts(
        model_name=model_name,
        model_path=args.model,
        output_dir=args.output_dir,
        vela_executable=args.vela_executable,
        accelerator_config=args.accelerator_config,
        config_path=args.config,
        system_config=args.system_config,
        memory_mode=args.memory_mode,
        optimise=args.optimise,
        tensor_allocator=args.tensor_allocator,
        extra_args=args.extra_arg,
    )
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
