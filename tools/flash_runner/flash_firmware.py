import argparse
import subprocess
from pathlib import Path


def firmware_image_path() -> Path:
    return Path("we2_image_gen_local") / "output_case1_sec_wlcsp" / "output.img"


def default_firmware_image_path() -> Path:
    return firmware_image_path()


def build_firmware_flash_command(
    port: str,
    image_path: str | Path,
    *,
    baudrate: int = 921600,
    protocol: str = "xmodem",
    auto_reset: bool = False,
    python_executable: str = "python",
    xmodem_script: str | Path = "xmodem/xmodem_send.py",
) -> list[str]:
    command = [
        python_executable,
        str(xmodem_script).replace("\\", "/"),
        "--port",
        port,
        "--baudrate",
        str(baudrate),
        "--protocol",
        protocol,
        "--file",
        Path(image_path).as_posix(),
    ]
    if auto_reset:
        command.append("--auto-reset")
    return command


def flash_firmware_image(
    port: str,
    image_path: str | Path | None = None,
    *,
    baudrate: int = 921600,
    protocol: str = "xmodem",
    auto_reset: bool = False,
    python_executable: str = "python",
    xmodem_script: str | Path = "xmodem/xmodem_send.py",
):
    image_path = image_path or firmware_image_path()
    command = build_firmware_flash_command(
        port=port,
        image_path=image_path,
        baudrate=baudrate,
        protocol=protocol,
        auto_reset=auto_reset,
        python_executable=python_executable,
        xmodem_script=xmodem_script,
    )
    return subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    parser.add_argument("--image", default=str(firmware_image_path()))
    parser.add_argument("--baudrate", default="921600")
    parser.add_argument("--protocol", default="xmodem")
    parser.add_argument("--auto-reset", action="store_true")
    parser.add_argument("--python-executable", default="python")
    parser.add_argument("--xmodem-script", default="xmodem/xmodem_send.py")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    flash_firmware_image(
        port=args.port,
        image_path=args.image,
        baudrate=int(args.baudrate, 0),
        protocol=args.protocol,
        auto_reset=args.auto_reset,
        python_executable=args.python_executable,
        xmodem_script=args.xmodem_script,
    )
    print(args.image)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
