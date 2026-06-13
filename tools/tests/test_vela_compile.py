from pathlib import Path
from unittest.mock import Mock, patch

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
    with patch("tools.model_converter.vela_compile.subprocess.run", return_value=completed) as run_mock:
        result = windows_path_to_wsl(Path("D:/example/model.tflite"))

    assert result == "/mnt/d/example/model.tflite"
    command = run_mock.call_args.args[0]
    assert command[:3] == ["wsl", "bash", "-lc"]
    assert "wslpath -a" in command[3]
