from pathlib import Path
from unittest.mock import Mock, patch
import subprocess

from tools.model_converter.vela_compile import build_vela_command
from tools.model_converter.vela_compile import compile_with_vela
from tools.model_converter.vela_compile import default_accelerator_config
from tools.model_converter.vela_compile import default_vela_config_path
from tools.model_converter.vela_compile import default_vela_executable
from tools.model_converter.vela_compile import default_vela_memory_mode
from tools.model_converter.vela_compile import default_vela_system_config
from tools.model_converter.vela_compile import main as vela_compile_main
from tools.model_converter.vela_compile import parse_vela_output
from tools.model_converter.vela_compile import windows_path_to_wsl


# --- Real Vela 5.1.0 output fixtures -----------------------------------------
#
# Captured from `vela 5.1.0` (binary at /home/hang22/.local/bin/vela) on
# 2026-06-13 against artifacts/yolo11n_od_192 and artifacts/yolov8n_od_192 with:
#   vela <int8.tflite> --output-dir /tmp/vela_probe \
#       --accelerator-config ethos-u55-256 --show-cpu-operations
#
# These are the hermetic regression fixtures for parse_vela_output. They must
# NOT be regenerated at test runtime (no vela/subprocess). The verbatim operator
# lines, the `CPU: <type> = <path> (outputs [...])` layout, the
# `CPU/NPU operators = N (P%)` summary spelling, and the trailing `NPU: <type>`
# lines (which must be excluded from cpu_fallback_ops) are all load-bearing.
#
# Trimmed to the structurally significant lines (warning banners, summary block,
# operator listings) to keep the fixture readable; every line that the parser
# could conceivably touch is preserved verbatim.

VELA_5_1_0_YOLO11N_STDOUT = """
Warning (supported operators) operator: CUSTOM, ofm: 'PartitionedCall:0'
Reason: Unsupported opType

Warning (supported operators) operator: CUSTOM, ofm: 'model_30/tf.concat_21/concat'
Reason: Unsupported opType
Warning: No configuration file specified. Using a default of ['/home/hang22/.local/lib/python3.10/site-packages/ethosu/config_files/Arm/vela.ini']. Compilation may be invalid or non-optimal.
Warning: No system configuration specified. Using a default of Ethos_U55_High_End_Embedded. Compilation may be invalid or non-optimal.
Warning: No memory mode specified. Using a default of Shared_Sram. Compilation may be invalid or non-optimal.

Network summary for yolo11n_od_192_int8
Accelerator configuration               Ethos_U55_256
System configuration             Ethos_U55_High_End_Embedded
Memory mode                               Shared_Sram
Accelerator clock                                 500 MHz
Design peak SRAM bandwidth                       3.73 GB/s

Total SRAM used                               2365.88 KiB

CPU operators = 10 (37.0%)
   CPU: Passthrough = model_30/tf.math.multiply_47/Mul (outputs [1, 24, 24, 128])
   CPU: Passthrough = model_30/tf.reshape_2/Reshape (outputs [1, 128, 2, 36])
   CPU: Passthrough = model_30/tf.math.add_54/Add (outputs [1, 6, 6, 128])
   CPU: Passthrough = model_30/tf.compat.v1.transpose_13/transpose/Transpose (outputs [1, 36, 2, 36])
   CPU: Passthrough = model_30/tf.reshape_8/Reshape (outputs [1, 6, 128, 6])
   CPU: Passthrough = model_30/tf.math.multiply_163/Mul (outputs [1, 6, 6, 256])
   CPU: Passthrough = model_30/tf.reshape_11/Reshape (outputs [1, 144, 1, 576])
   CPU: Passthrough = model_30/tf.reshape_14/Reshape (outputs [1, 144, 1, 144])
   CPU: Passthrough = model_30/tf.concat_21/concat (outputs [1, 144, 1, 756])
   CPU: Passthrough = PartitionedCall:0 (outputs [1, 84, 1, 756])
NPU operators = 17 (63.0%)
   NPU: Transpose = #96 (outputs [1, 6, 256, 6])
   NPU: MatMul = model_30/tf.linalg.matmul_1/MatMul;model_30/tf.compat.v1.transpose_6/transpose (outputs [1, 36, 2, 36])
   NPU: Transpose = model_30/tf.compat.v1.transpose_39/transpose;model_30/tf.compat.v1.transpose_38/transpose/perm;model_30/tf.compat.v1.transpose_38/transpose; (outputs [1, 756, 4, 16])

Average SRAM bandwidth                           2.16 GB/s
Neural network macs                            248832 MACs/batch
""".strip()


VELA_5_1_0_YOLOV8N_STDOUT = """
Warning (supported operators) operator: CUSTOM, ofm: 'PartitionedCall:0'
Reason: Unsupported opType
Warning: No configuration file specified. Using a default of ['/home/hang22/.local/lib/python3.10/site-packages/ethosu/config_files/Arm/vela.ini']. Compilation may be invalid or non-optimal.
Warning: No system configuration specified. Using a default of Ethos_U55_High_End_Embedded. Compilation may be invalid or non-optimal.
Warning: No memory mode specified. Using a default of Shared_Sram. Compilation may be invalid or non-optimal.

Network summary for yolov8n_od_192_int8
Accelerator configuration               Ethos_U55_256
System configuration             Ethos_U55_High_End_Embedded
Memory mode                               Shared_Sram
Accelerator clock                                 500 MHz
Design peak SRAM bandwidth                       3.73 GB/s

Total SRAM used                               2222.44 KiB

CPU operators = 2 (66.7%)
   CPU: Passthrough = PartitionedCall:0 (outputs [1, 756, 1, 80])
   CPU: Passthrough = PartitionedCall:1 (outputs [1, 4, 1, 756])
NPU operators = 1 (33.3%)
   NPU: Transpose = model_9/tf.compat.v1.transpose_9/transpose (outputs [1, 756, 4, 16])

Average SRAM bandwidth                           3.81 GB/s
Neural network macs                                 0 MACs/batch
""".strip()


# --- Real-output regression tests --------------------------------------------


def test_parse_vela_output_yolo11n_real_vela_5_1_0() -> None:
    """yolo11n_od_192_int8.tflite via real Vela 5.1.0: 10 CPU / 17 NPU ops.

    Hand-read from VELA_5_1_0_YOLO11N_STDOUT:
      - summary line `CPU operators = 10 (37.0%)`  -> cpu_ops 10
      - summary line `NPU operators = 17 (63.0%)`  -> npu_ops 17
      - utilization 17/(10+17) = 63.0%
      - all 10 CPU fallback lines are `CPU: Passthrough = <path>`, so the
        order-preserving dedup collapses them to a single `Passthrough`.
      - the `NPU: Transpose/MatMul = ...` lines must NOT leak into cpu_fallback.
    """
    summary = parse_vela_output(VELA_5_1_0_YOLO11N_STDOUT)

    assert summary["cpu_ops"] == 10
    assert summary["npu_ops"] == 17
    assert summary["npu_utilization_pct"] == 63.0
    assert summary["cpu_fallback_ops"] == ["Passthrough"]


def test_parse_vela_output_yolov8n_real_vela_5_1_0() -> None:
    """yolov8n_od_192_int8.tflite via real Vela 5.1.0: 2 CPU / 1 NPU ops.

    Hand-read from VELA_5_1_0_YOLOV8N_STDOUT:
      - `CPU operators = 2 (66.7%)`  -> cpu_ops 2
      - `NPU operators = 1 (33.3%)`  -> npu_ops 1
      - utilization 1/(2+1) = 33.3% (rounded to 1 decimal)
      - 2 Passthrough fallback lines dedup to `['Passthrough']`.
      - `NPU: Transpose = ...` excluded.
    """
    summary = parse_vela_output(VELA_5_1_0_YOLOV8N_STDOUT)

    assert summary["cpu_ops"] == 2
    assert summary["npu_ops"] == 1
    assert summary["npu_utilization_pct"] == 33.3
    assert summary["cpu_fallback_ops"] == ["Passthrough"]


def test_parse_vela_output_keeps_fallback_order_and_dedup() -> None:
    """Mixed fallback types must be deduped while preserving first-seen order.

    Synthetic but mirrors the real `CPU: <type> = <path>` shape, including the
    passthrough-on-vela case where the same type repeats many times.
    """
    sample = """
CPU operators = 4 (50.0%)
   CPU: Passthrough = model_30/tf.math.multiply_47/Mul (outputs [1, 24, 24, 128])
   CPU: Reshape = model_30/tf.reshape_2/Reshape (outputs [1, 128, 2, 36])
   CPU: Passthrough = model_30/tf.math.add_54/Add (outputs [1, 6, 6, 128])
   CPU: Reshape = model_30/tf.reshape_8/Reshape (outputs [1, 6, 128, 6])
NPU operators = 4 (50.0%)
    """.strip()

    summary = parse_vela_output(sample)

    assert summary["cpu_ops"] == 4
    assert summary["npu_ops"] == 4
    assert summary["npu_utilization_pct"] == 50.0
    assert summary["cpu_fallback_ops"] == ["Passthrough", "Reshape"]


# --- Legacy format coverage (older vela spelling; still supported) -----------


def test_parse_vela_output() -> None:
    sample = """
Network summary
Accelerator configuration                 Ethos_U55_256
CPU operations                            2
NPU operations                            38
Average SRAM used                         912.00 KiB
CPU operation breakdown
  RESHAPE: 1
  TRANSPOSE: 1
    """.strip()

    summary = parse_vela_output(sample)

    assert summary["npu_ops"] == 38
    assert summary["cpu_ops"] == 2
    assert summary["npu_utilization_pct"] == 95.0
    assert summary["cpu_fallback_ops"] == ["RESHAPE", "TRANSPOSE"]


def test_parse_vela_output_operator_style() -> None:
    sample = """
Operator Coverage
CPU operators = 10 (37.0%)
NPU operators = 17 (63.0%)
    CPU: Passthrough = model_30/tf.math.multiply_47/Mul
    CPU: Reshape = model_30/reshape_1/Reshape
    CPU: Passthrough = model_30/tf.math.multiply_52/Mul
    """.strip()

    summary = parse_vela_output(sample)

    assert summary["npu_ops"] == 17
    assert summary["cpu_ops"] == 10
    assert summary["npu_utilization_pct"] == 63.0
    assert summary["cpu_fallback_ops"] == ["Passthrough", "Reshape"]


def test_windows_path_to_wsl_uses_bash_bridge() -> None:
    completed = Mock(stdout="/mnt/d/example/model.tflite\n")
    with (
        patch.dict(
            "tools.model_converter.vela_compile.os.environ",
            {"WSL_DISTRO_NAME": "", "WSL_INTEROP": ""},
            clear=False,
        ),
        patch("tools.model_converter.vela_compile.subprocess.run", return_value=completed) as run_mock,
    ):
        result = windows_path_to_wsl(Path("D:/example/model.tflite"))

    assert result == "/mnt/d/example/model.tflite"
    command = run_mock.call_args.args[0]
    assert command[:3] == ["wsl", "bash", "-lc"]
    assert "wslpath -a" in command[3]


def test_windows_path_to_wsl_returns_native_path_inside_wsl() -> None:
    with patch.dict(
        "tools.model_converter.vela_compile.os.environ",
        {"WSL_DISTRO_NAME": "Ubuntu-22.04"},
        clear=False,
    ):
        result = windows_path_to_wsl("/tmp/example/model.tflite")

    assert result == str(Path("/tmp/example/model.tflite").resolve())


def test_build_vela_command_uses_native_invocation_inside_wsl() -> None:
    with (
        patch.dict(
            "tools.model_converter.vela_compile.os.environ",
            {"WSL_DISTRO_NAME": "Ubuntu-22.04"},
            clear=False,
        ),
        patch("tools.model_converter.vela_compile.default_vela_config_path", return_value="/repo/configs/himax_vela.ini"),
        patch("tools.model_converter.vela_compile.default_vela_system_config", return_value="My_Sys_Cfg"),
        patch("tools.model_converter.vela_compile.default_vela_memory_mode", return_value="My_Mem_Mode_Parent"),
    ):
        command = build_vela_command(
            model_path="/tmp/example/model.tflite",
            output_dir="/tmp/example/out",
            vela_executable="/home/hang22/.local/bin/vela",
        )

    assert command == [
        "/home/hang22/.local/bin/vela",
        str(Path("/tmp/example/model.tflite").resolve()),
        "--output-dir",
        str(Path("/tmp/example/out").resolve()),
        "--accelerator-config",
        "ethos-u55-64",
        "--show-cpu-operations",
        "--config",
        str(Path("/repo/configs/himax_vela.ini").resolve()),
        "--system-config",
        "My_Sys_Cfg",
        "--memory-mode",
        "My_Mem_Mode_Parent",
    ]


def test_build_vela_command_supports_python_runner_and_optimise_override() -> None:
    with (
        patch.dict(
            "tools.model_converter.vela_compile.os.environ",
            {"WSL_DISTRO_NAME": "Ubuntu-22.04"},
            clear=False,
        ),
        patch("tools.model_converter.vela_compile.default_vela_config_path", return_value="/repo/configs/himax_vela.ini"),
        patch("tools.model_converter.vela_compile.default_vela_system_config", return_value="My_Sys_Cfg"),
        patch("tools.model_converter.vela_compile.default_vela_memory_mode", return_value="My_Mem_Mode_Parent"),
    ):
        command = build_vela_command(
            model_path="/tmp/example/model.tflite",
            output_dir="/tmp/example/out",
            vela_executable="python3",
            vela_executable_args=["artifacts_debug/run_vela450.py"],
            optimise="Size",
        )

    assert command == [
        "python3",
        str((Path("artifacts_debug") / "run_vela450.py").resolve()),
        str(Path("/tmp/example/model.tflite").resolve()),
        "--output-dir",
        str(Path("/tmp/example/out").resolve()),
        "--accelerator-config",
        "ethos-u55-64",
        "--show-cpu-operations",
        "--config",
        str(Path("/repo/configs/himax_vela.ini").resolve()),
        "--system-config",
        "My_Sys_Cfg",
        "--memory-mode",
        "My_Mem_Mode_Parent",
        "--optimise",
        "Size",
    ]


def test_vela_compile_cli_forwards_size_options(tmp_path: Path) -> None:
    model_path = tmp_path / "model.tflite"
    model_path.write_bytes(b"tfl3")

    with (
        patch(
            "tools.model_converter.vela_compile.sys.argv",
            [
                "vela_compile.py",
                str(model_path),
                "--model-name",
                "probe",
                "--output-dir",
                str(tmp_path / "vela"),
                "--optimise",
                "Size",
                "--tensor-allocator",
                "HillClimb",
                "--extra-arg=--verbose-operators",
            ],
        ),
        patch("tools.model_converter.vela_compile.write_vela_artifacts", return_value=tmp_path / "vela" / "probe.vela_info.json") as write_mock,
    ):
        assert vela_compile_main() == 0

    kwargs = write_mock.call_args.kwargs
    assert kwargs["optimise"] == "Size"
    assert kwargs["tensor_allocator"] == "HillClimb"
    assert kwargs["extra_args"] == ["--verbose-operators"]


def test_default_accelerator_config_defaults_to_hx6538_target() -> None:
    with patch.dict("tools.model_converter.vela_compile.os.environ", {}, clear=True):
        assert default_accelerator_config() == "ethos-u55-64"


def test_default_accelerator_config_honors_explicit_env_override() -> None:
    with patch.dict(
        "tools.model_converter.vela_compile.os.environ",
        {"VELA_ACCELERATOR_CONFIG": "ethos-u55-128"},
        clear=True,
    ):
        assert default_accelerator_config() == "ethos-u55-128"


def test_default_vela_config_path_defaults_to_repo_himax_config() -> None:
    with patch.dict("tools.model_converter.vela_compile.os.environ", {}, clear=True):
        assert default_vela_config_path() == str(Path("configs/himax_vela.ini").resolve())


def test_default_vela_config_path_honors_explicit_env_override() -> None:
    with patch.dict(
        "tools.model_converter.vela_compile.os.environ",
        {"VELA_CONFIG": "/custom/himax_vela.ini"},
        clear=True,
    ):
        assert default_vela_config_path() == "/custom/himax_vela.ini"


def test_default_vela_system_and_memory_mode_follow_repo_himax_defaults() -> None:
    with patch.dict("tools.model_converter.vela_compile.os.environ", {}, clear=True):
        assert default_vela_system_config() == "My_Sys_Cfg"
        assert default_vela_memory_mode() == "My_Mem_Mode_Parent"


def test_default_vela_system_and_memory_mode_do_not_guess_for_custom_config_without_names() -> None:
    with patch.dict(
        "tools.model_converter.vela_compile.os.environ",
        {"VELA_CONFIG": "/custom/board.ini"},
        clear=True,
    ):
        assert default_vela_system_config() is None
        assert default_vela_memory_mode() is None


def test_default_vela_system_and_memory_mode_honor_explicit_env_override() -> None:
    with patch.dict(
        "tools.model_converter.vela_compile.os.environ",
        {
            "VELA_SYSTEM_CONFIG": "CustomSys",
            "VELA_MEMORY_MODE": "CustomMem",
        },
        clear=True,
    ):
        assert default_vela_system_config() == "CustomSys"
        assert default_vela_memory_mode() == "CustomMem"


def test_default_vela_executable_prefers_miniforge_vela_env() -> None:
    with (
        patch.dict("tools.model_converter.vela_compile.os.environ", {}, clear=True),
        patch("tools.model_converter.vela_compile.Path.home", return_value=Path("/home/tester")),
        patch("tools.model_converter.vela_compile.Path.exists", autospec=True) as exists_mock,
    ):
        exists_mock.side_effect = lambda self: str(self).replace("\\", "/") == "/home/tester/miniforge3/envs/vela/bin/vela"
        assert default_vela_executable().replace("\\", "/") == "/home/tester/miniforge3/envs/vela/bin/vela"


def test_default_vela_executable_honors_explicit_env_override() -> None:
    with patch.dict(
        "tools.model_converter.vela_compile.os.environ",
        {"VELA_EXECUTABLE": "/custom/vela"},
        clear=True,
    ):
        assert default_vela_executable() == "/custom/vela"


def test_compile_with_vela_surfaces_process_output_on_failure(tmp_path: Path) -> None:
    error = subprocess.CalledProcessError(
        returncode=1,
        cmd=["vela", "model.tflite"],
        output="stdout details",
        stderr="stderr details",
    )
    with (
        patch.dict(
            "tools.model_converter.vela_compile.os.environ",
            {"WSL_DISTRO_NAME": "Ubuntu-22.04"},
            clear=False,
        ),
        patch(
            "tools.model_converter.vela_compile.subprocess.run",
            side_effect=error,
        ),
    ):
        try:
            compile_with_vela(
                model_path=tmp_path / "model.tflite",
                output_dir=tmp_path / "out",
                vela_executable="/custom/vela",
            )
        except RuntimeError as exc:
            message = str(exc)
        else:
            raise AssertionError("expected compile_with_vela to raise RuntimeError")

    assert "stdout details" in message
    assert "stderr details" in message
