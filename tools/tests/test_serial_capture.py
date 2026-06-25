import sys
import types
from pathlib import Path

import pytest

from tools.flash_runner.serial_capture import capture_json_objects
from tools.flash_runner.serial_capture import extract_json_objects


def test_extract_json_objects() -> None:
    lines = [
        "boot ok",
        '{"benchmark":{"device":"himax_hx6538","firmware":"model_benchmark_v1"},"models":[',
        '{"name":"yolo11n_od_192","task":"object_detection","model_size_bytes":4532736,',
        '"arena_used_bytes":1048576,"latency_ms":{"avg":85.3,"min":82.1,"max":91.7},"runs":10,"status":"ok"},',
        '{"name":"yolov8n_od_192","task":"object_detection","model_size_bytes":3984588,',
        '"arena_used_bytes":983040,"latency_ms":{"avg":79.5,"min":77.2,"max":83.9},"runs":10,"status":"ok"}]}',
        "done",
    ]

    payloads = extract_json_objects(lines)

    assert payloads[0]["benchmark"]["device"] == "himax_hx6538"
    assert len(payloads[0]["models"]) == 2
    assert payloads[0]["models"][1]["latency_ms"]["avg"] == 79.5


def test_skips_invalid_json_fragments() -> None:
    lines = [
        '{"benchmark":{"device":"broken"},"models":[not-json]}',
        '{"benchmark":{"device":"himax_hx6538"},"models":[{"name":"ok","status":"ok"}]}',
    ]

    payloads = extract_json_objects(lines)

    assert len(payloads) == 1
    assert payloads[0]["models"][0]["name"] == "ok"


def test_extract_json_objects_tolerates_trailing_serial_noise() -> None:
    lines = [
        'junk before payload {"benchmark":{"device":"himax_hx6538"},"models":[{"name":"ok","status":"ok"}]}\ufffd\ufffd\ufffd',
    ]

    payloads = extract_json_objects(lines)

    assert len(payloads) == 1
    assert payloads[0]["benchmark"]["device"] == "himax_hx6538"
    assert payloads[0]["models"][0]["name"] == "ok"


def test_extract_json_objects_tolerates_ethosu_error_lines_inside_payload() -> None:
    lines = [
        '{"benchmark":{"device":"himax_hx6538","firmware":"model_benchmark_v1"},"models":[E: NPU config mismatch. npu.macs_per_cc=6, optimizer.macs_per_cc=8 (ethosu_device_u55_u65.c:337)',
        'E: Failed to invoke inference. (ethosu_driver.c:736)',
        'Node ethos-u (number 0) failed to invoke with status 1',
        '{"name":"mobilenetv3small_cls_224","task":"classification","model_size_bytes":2506400,"arena_used_bytes":352804,"latency_ms":{"avg":0,"min":0,"max":0},"runs":10,"status":"invoke_failed"}]}',
    ]

    payloads = extract_json_objects(lines)

    assert len(payloads) == 1
    assert payloads[0]["benchmark"]["device"] == "himax_hx6538"
    assert payloads[0]["models"][0]["name"] == "mobilenetv3small_cls_224"
    assert payloads[0]["models"][0]["status"] == "invoke_failed"


def test_extract_json_objects_preserves_sub_ms_latency_values() -> None:
    lines = [
        '{"benchmark":{"device":"himax_hx6538","firmware":"model_benchmark_v1"},"models":[',
        '{"name":"mobilenetv3small_cls_224","task":"classification","model_size_bytes":2506400,',
        '"arena_used_bytes":352804,"latency_ms":{"avg":0.612,"min":0.601,"max":0.644},"runs":10,"status":"ok"}]}',
    ]

    payloads = extract_json_objects(lines)

    assert len(payloads) == 1
    assert payloads[0]["models"][0]["latency_ms"]["avg"] == pytest.approx(0.612)
    assert payloads[0]["models"][0]["latency_ms"]["min"] == pytest.approx(0.601)
    assert payloads[0]["models"][0]["latency_ms"]["max"] == pytest.approx(0.644)


def test_capture_json_objects_reads_until_first_complete_payload(monkeypatch, tmp_path: Path) -> None:
    transcript = [
        b"booting...\r\n",
        b'{"benchmark":{"device":"himax_hx6538"},"models":[{"name":"ok","status":"ok"}]}\xef\xbf\xbd\r\n',
    ]

    class _FakeSerialPort:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs
            self._lines = iter(transcript)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def readline(self) -> bytes:
            return next(self._lines, b"")

    fake_serial_module = types.SimpleNamespace(Serial=_FakeSerialPort)
    monkeypatch.setitem(sys.modules, "serial", fake_serial_module)

    log_path = tmp_path / "capture.log"
    payloads = capture_json_objects("COM7", baudrate=921600, timeout=1.0, log_path=log_path)

    assert len(payloads) == 1
    assert payloads[0]["models"][0]["name"] == "ok"
    assert "booting..." in log_path.read_text(encoding="utf-8")
    assert "himax_hx6538" in log_path.read_text(encoding="utf-8")


def test_capture_json_objects_uses_serial_url_for_socket_ports(monkeypatch) -> None:
    transcript = [
        b'{"benchmark":{"device":"himax_hx6538"},"models":[{"name":"socket-ok","status":"ok"}]}\r\n',
    ]
    opened_urls: list[tuple[str, int, float]] = []

    class _FakeSerialPort:
        def __init__(self, *args, **kwargs) -> None:
            self._lines = iter(transcript)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def readline(self) -> bytes:
            return next(self._lines, b"")

    def _serial_for_url(port: str, *, baudrate: int, timeout: float):
        opened_urls.append((port, baudrate, timeout))
        return _FakeSerialPort()

    fake_serial_module = types.SimpleNamespace(Serial=None, serial_for_url=_serial_for_url)
    monkeypatch.setitem(sys.modules, "serial", fake_serial_module)

    payloads = capture_json_objects("socket://127.0.0.1:57005", baudrate=921600, timeout=1.0)

    assert payloads[0]["models"][0]["name"] == "socket-ok"
    assert opened_urls == [("socket://127.0.0.1:57005", 921600, 0.2)]


def test_capture_json_objects_times_out_without_complete_payload(monkeypatch) -> None:
    class _FakeSerialPort:
        def __init__(self, *args, **kwargs) -> None:
            self._lines = iter([b"boot\r\n", b"still waiting\r\n", b""])

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def readline(self) -> bytes:
            return next(self._lines, b"")

    fake_serial_module = types.SimpleNamespace(Serial=_FakeSerialPort)
    monkeypatch.setitem(sys.modules, "serial", fake_serial_module)

    ticks = iter([0.0, 0.2, 0.6, 1.1, 1.2])
    monkeypatch.setattr("tools.flash_runner.serial_capture.time.monotonic", lambda: next(ticks))

    with pytest.raises(TimeoutError, match="COM7"):
        capture_json_objects("COM7", baudrate=921600, timeout=1.0)
