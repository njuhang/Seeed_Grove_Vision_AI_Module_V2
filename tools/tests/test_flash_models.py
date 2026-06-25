import json
from pathlib import Path
from unittest.mock import Mock
from unittest.mock import patch

from tools.flash_runner.flash_models import build_flash_command
from tools.flash_runner.flash_models import build_model_args
from tools.flash_runner.flash_models import flash_batch_from_plan
from tools.flash_runner.flash_models import flash_model_batch
from tools.flash_runner.flash_models import parse_args


def test_build_model_args() -> None:
    args = build_model_args(
        [
            ("artifacts/yolo11n_od_192/yolo11n_od_192_vela.tflite", 0x00400000),
            ("artifacts/yolov8n_od_192/yolov8n_od_192_vela.tflite", 0x00600000),
        ]
    )

    assert args == [
        "--model",
        "artifacts/yolo11n_od_192/yolo11n_od_192_vela.tflite 0x400000 0x00000",
        "--model",
        "artifacts/yolov8n_od_192/yolov8n_od_192_vela.tflite 0x600000 0x00000",
    ]


def test_build_model_args_translates_wsl_paths_for_windows_python() -> None:
    args = build_model_args(
        [
            ("/mnt/d/code/repo/artifacts/yolov5n_od_192/yolov5n_od_192_vela.tflite", 0x00C00000),
        ]
    )

    assert args == [
        "--model",
        "D:/code/repo/artifacts/yolov5n_od_192/yolov5n_od_192_vela.tflite 0xc00000 0x00000",
    ]


def test_build_model_args_splits_large_models_into_chunk_offsets(tmp_path: Path) -> None:
    model_path = tmp_path / "yolov8n_od_192_vela.tflite"
    model_path.write_bytes(b"abcdefghij")

    args = build_model_args(
        [(str(model_path), 0x00600000)],
        model_chunk_size=4,
        chunk_root=tmp_path / "chunks",
    )

    assert args == [
        "--model",
        f"{(tmp_path / 'chunks' / 'yolov8n_od_192_vela' / 'yolov8n_od_192_vela_chunk_00.bin').as_posix()} 0x600000 0x00000",
        "--model",
        f"{(tmp_path / 'chunks' / 'yolov8n_od_192_vela' / 'yolov8n_od_192_vela_chunk_01.bin').as_posix()} 0x600000 0x00004",
        "--model",
        f"{(tmp_path / 'chunks' / 'yolov8n_od_192_vela' / 'yolov8n_od_192_vela_chunk_02.bin').as_posix()} 0x600000 0x00008",
    ]
    assert (tmp_path / "chunks" / "yolov8n_od_192_vela" / "yolov8n_od_192_vela_chunk_00.bin").read_bytes() == b"abcd"
    assert (tmp_path / "chunks" / "yolov8n_od_192_vela" / "yolov8n_od_192_vela_chunk_01.bin").read_bytes() == b"efgh"
    assert (tmp_path / "chunks" / "yolov8n_od_192_vela" / "yolov8n_od_192_vela_chunk_02.bin").read_bytes() == b"ij"


def test_build_flash_command_includes_model_table_first() -> None:
    command = build_flash_command(
        port="COM7",
        table_path=Path("artifacts/model_table_batch_0.bin"),
        models=[
            ("artifacts/yolo11n_od_192/yolo11n_od_192_vela.tflite", 0x00400000),
            ("artifacts/yolov8n_od_192/yolov8n_od_192_vela.tflite", 0x00600000),
        ],
    )

    assert command[:6] == [
        "python",
        "xmodem/xmodem_send.py",
        "--port",
        "COM7",
        "--baudrate",
        "921600",
    ]
    table_idx = command.index("--model")
    assert command[table_idx + 1] == "artifacts/model_table_batch_0.bin 0x200000 0x00000"
    first_model_idx = command.index("--model", table_idx + 2)
    assert command[first_model_idx + 1] == "artifacts/yolo11n_od_192/yolo11n_od_192_vela.tflite 0x400000 0x00000"
    assert table_idx < first_model_idx


def test_build_flash_command_supports_auto_reset() -> None:
    command = build_flash_command(
        port="COM7",
        table_path=Path("artifacts/model_table_batch_0.bin"),
        models=[("artifacts/yolo11n_od_192/yolo11n_od_192_vela.tflite", 0x00400000)],
        auto_reset=True,
    )

    assert "--auto-reset" in command


def test_flash_model_batch_runs_subprocess_with_built_command() -> None:
    completed = Mock()
    completed.returncode = 0

    with patch("tools.flash_runner.flash_models.subprocess.run", return_value=completed) as run_mock:
        result = flash_model_batch(
            port="COM7",
            table_path=Path("artifacts/model_table_batch_0.bin"),
            models=[("artifacts/yolo11n_od_192/yolo11n_od_192_vela.tflite", 0x00400000)],
        )

    assert result is completed
    run_mock.assert_called_once()
    assert run_mock.call_args.kwargs["check"] is True


def test_flash_model_batch_uses_chunk_root_next_to_model_table(tmp_path: Path) -> None:
    completed = Mock()
    completed.returncode = 0
    table_path = tmp_path / "model_table_batch_0.bin"
    table_path.write_bytes(b"table")
    model_path = tmp_path / "yolov8n_od_192_vela.tflite"
    model_path.write_bytes(b"abcdefghij")

    with patch("tools.flash_runner.flash_models.subprocess.run", return_value=completed) as run_mock:
        flash_model_batch(
            port="COM7",
            table_path=table_path,
            models=[(str(model_path), 0x00600000)],
            model_chunk_size=4,
        )

    command = run_mock.call_args.args[0]
    chunk_0 = str(tmp_path / "model_chunks" / "yolov8n_od_192_vela" / "yolov8n_od_192_vela_chunk_00.bin").replace("\\", "/")
    chunk_1 = str(tmp_path / "model_chunks" / "yolov8n_od_192_vela" / "yolov8n_od_192_vela_chunk_01.bin").replace("\\", "/")
    chunk_2 = str(tmp_path / "model_chunks" / "yolov8n_od_192_vela" / "yolov8n_od_192_vela_chunk_02.bin").replace("\\", "/")
    assert any(arg.startswith(chunk_0) for arg in command)
    assert any(arg.startswith(chunk_1) for arg in command)
    assert any(arg.startswith(chunk_2) for arg in command)


def test_flash_batch_from_plan_selects_requested_batch(tmp_path: Path) -> None:
    plan_path = tmp_path / "benchmark_plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "batches": [
                    {
                        "batch_index": 0,
                        "table_path": str(tmp_path / "model_table_batch_0.bin"),
                        "models": [
                            {
                                "model_path": "artifacts/yolo11n_od_192/yolo11n_od_192_vela.tflite",
                                "flash_address": "0x400000",
                            }
                        ],
                    },
                    {
                        "batch_index": 1,
                        "table_path": str(tmp_path / "model_table_batch_1.bin"),
                        "models": [
                            {
                                "model_path": "artifacts/yolov8n_od_192/yolov8n_od_192_vela.tflite",
                                "flash_address": "0x600000",
                            }
                        ],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    with patch("tools.flash_runner.flash_models.flash_model_batch") as flash_mock:
        flash_batch_from_plan(plan_path, batch_index=1, port="COM7")

    flash_mock.assert_called_once_with(
        port="COM7",
        table_path=str(tmp_path / "model_table_batch_1.bin"),
        models=[("artifacts/yolov8n_od_192/yolov8n_od_192_vela.tflite", 0x600000)],
        baudrate=921600,
        protocol="xmodem",
        auto_reset=False,
        python_executable="python",
        xmodem_script="xmodem/xmodem_send.py",
        model_chunk_size=None,
    )


def test_parse_args_requires_batch_and_plan(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "flash_models.py",
            "--port",
            "COM7",
            "--plan",
            "artifacts/benchmark_plan.json",
            "--batch",
            "1",
        ],
    )

    args = parse_args()

    assert args.port == "COM7"
    assert args.plan == "artifacts/benchmark_plan.json"
    assert args.batch == 1


def test_parse_args_supports_auto_reset(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "flash_models.py",
            "--port",
            "COM7",
            "--batch",
            "1",
            "--auto-reset",
        ],
    )

    args = parse_args()

    assert args.auto_reset is True


def test_parse_args_supports_model_chunk_size(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "flash_models.py",
            "--port",
            "COM7",
            "--batch",
            "1",
            "--model-chunk-size",
            "0x10000",
        ],
    )

    args = parse_args()

    assert args.model_chunk_size == "0x10000"
