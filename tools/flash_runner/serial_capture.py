import json
import re
import time
from json import JSONDecodeError
from pathlib import Path


def _strip_known_runtime_noise(lines: list[str]) -> list[str]:
    noise_patterns = (
        r"E: [^\r\n]*",
        r"Node ethos-u [^\r\n]*",
    )
    cleaned: list[str] = []
    for raw_line in lines:
        line = raw_line.lstrip("\x00")
        for pattern in noise_patterns:
            line = re.sub(pattern, "", line)
        if line.strip():
            cleaned.append(line)
    return cleaned


def _iter_json_texts(lines: list[str]):
    buffer: list[str] = []
    brace_depth = 0
    in_string = False
    escaping = False

    for line in lines:
        for char in line:
            if not buffer:
                if char != "{":
                    continue
                buffer.append(char)
                brace_depth = 1
                in_string = False
                escaping = False
                continue

            buffer.append(char)

            if escaping:
                escaping = False
                continue

            if char == "\\" and in_string:
                escaping = True
                continue

            if char == '"':
                in_string = not in_string
                continue

            if in_string:
                continue

            if char == "{":
                brace_depth += 1
            elif char == "}":
                brace_depth -= 1
                if brace_depth == 0:
                    yield "".join(buffer)
                    buffer = []


def extract_json_objects(lines: list[str]) -> list[dict]:
    payloads: list[dict] = []

    for json_text in _iter_json_texts(_strip_known_runtime_noise(lines)):
        try:
            payloads.append(json.loads(json_text))
        except JSONDecodeError:
            continue
    return payloads


def _open_serial_endpoint(serial_module, port: str, baudrate: int, timeout: float):
    if "://" in port:
        return serial_module.serial_for_url(port, baudrate=baudrate, timeout=timeout)
    return serial_module.Serial(port, baudrate=baudrate, timeout=timeout)


def capture_json_objects(
    port: str,
    baudrate: int = 921600,
    timeout: float = 60.0,
    log_path: str | Path | None = None,
    *,
    read_timeout: float = 0.2,
) -> list[dict]:
    import serial

    deadline = time.monotonic() + timeout
    lines: list[str] = []
    log_file = Path(log_path) if log_path is not None else None

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)

    with _open_serial_endpoint(serial, port, baudrate, read_timeout) as serial_port:
        while time.monotonic() < deadline:
            raw = serial_port.readline()
            if not raw:
                continue

            decoded = raw.decode("utf-8", errors="ignore").replace("\x00", "")
            lines.append(decoded)

            if log_file is not None:
                with log_file.open("a", encoding="utf-8") as handle:
                    handle.write(decoded)

            payloads = extract_json_objects(lines)
            if payloads:
                return payloads

    raise TimeoutError(f"timed out waiting for benchmark JSON on {port}")
