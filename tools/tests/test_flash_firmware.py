from pathlib import Path
from unittest.mock import Mock
from unittest.mock import patch

from tools.flash_runner.flash_firmware import default_firmware_image_path
from tools.flash_runner.flash_firmware import build_firmware_flash_command
from tools.flash_runner.flash_firmware import parse_args
from tools.flash_runner.flash_firmware import firmware_image_path
from tools.flash_runner.flash_firmware import flash_firmware_image


def test_firmware_image_path() -> None:
    assert firmware_image_path() == Path("we2_image_gen_local") / "output_case1_sec_wlcsp" / "output.img"
    assert default_firmware_image_path() == firmware_image_path()


def test_build_firmware_flash_command() -> None:
    command = build_firmware_flash_command(
        port="COM7",
        image_path=Path("we2_image_gen_local/output_case1_sec_wlcsp/output.img"),
    )

    assert command == [
        "python",
        "xmodem/xmodem_send.py",
        "--port",
        "COM7",
        "--baudrate",
        "921600",
        "--protocol",
        "xmodem",
        "--file",
        "we2_image_gen_local/output_case1_sec_wlcsp/output.img",
    ]


def test_build_firmware_flash_command_supports_auto_reset() -> None:
    command = build_firmware_flash_command(
        port="COM7",
        image_path=Path("we2_image_gen_local/output_case1_sec_wlcsp/output.img"),
        auto_reset=True,
    )

    assert command[-1] == "--auto-reset"


def test_flash_firmware_image_runs_subprocess() -> None:
    completed = Mock()
    completed.returncode = 0

    with patch("tools.flash_runner.flash_firmware.subprocess.run", return_value=completed) as run_mock:
        result = flash_firmware_image(
            port="COM7",
            image_path=Path("we2_image_gen_local/output_case1_sec_wlcsp/output.img"),
        )

    assert result is completed
    run_mock.assert_called_once()
    assert run_mock.call_args.kwargs["check"] is True


def test_parse_args_uses_default_firmware_image(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "flash_firmware.py",
            "--port",
            "COM7",
        ],
    )

    args = parse_args()

    assert args.port == "COM7"
    assert args.image == str(firmware_image_path())


def test_parse_args_supports_auto_reset(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "flash_firmware.py",
            "--port",
            "COM7",
            "--auto-reset",
        ],
    )

    args = parse_args()

    assert args.auto_reset is True
