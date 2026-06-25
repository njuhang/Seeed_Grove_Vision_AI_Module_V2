import importlib.util
import itertools
from pathlib import Path
from types import SimpleNamespace
import sys
import types

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
XMODEM_SEND_PATH = REPO_ROOT / "xmodem" / "xmodem_send.py"


def _load_xmodem_send_module():
    fake_serial = types.ModuleType("serial")

    class _FakeSerialPort:
        def __init__(self, *args, **kwargs):
            self.port = None
            self.timeout = None
            self.baudrate = None
            self.bytesize = None
            self.stopbits = None
            self.xonxoff = None
            self.rtscts = None
            self.parity = None

        def open(self):
            return None

        def flushInput(self):
            return None

        def flushOutput(self):
            return None

        def close(self):
            return None

    fake_serial.Serial = _FakeSerialPort
    fake_serial.serial_for_url = lambda *args, **kwargs: _FakeSerialPort(*args, **kwargs)
    fake_serial.SerialException = Exception
    fake_serial.EIGHTBITS = 8
    fake_serial.STOPBITS_ONE = 1
    fake_serial.PARITY_NONE = "N"

    fake_xmodem = types.ModuleType("xmodem")

    class _FakeXMODEM:
        def __init__(self, *args, **kwargs):
            pass

        def send(self, *args, **kwargs):
            return True

    fake_xmodem.XMODEM = _FakeXMODEM

    sys.modules.setdefault("serial", fake_serial)
    sys.modules.setdefault("xmodem", fake_xmodem)

    spec = importlib.util.spec_from_file_location("xmodem_send_under_test", XMODEM_SEND_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _FakeSerial:
    def __init__(self, responses: list[bytes]) -> None:
        self.responses = iter(responses)
        self.timeout = 60.0
        self.writes: list[bytes] = []
        self.flush_calls = 0
        self.readline_calls = 0

    def readline(self) -> bytes:
        self.readline_calls += 1
        return next(self.responses, b"")

    def write(self, data: bytes) -> int:
        self.writes.append(data)
        return len(data)

    def flush(self) -> None:
        self.flush_calls += 1


class _FlakyWriteSerial(_FakeSerial):
    def __init__(self, responses: list[bytes] | None = None) -> None:
        super().__init__(responses or [])
        self.write_attempts = 0

    def write(self, data: bytes) -> int:
        self.write_attempts += 1
        if self.write_attempts == 1:
            raise PermissionError(13, "Access is denied")
        return super().write(data)


def test_wait_for_xmodem_ready_retries_reset_until_ready(monkeypatch) -> None:
    module = _load_xmodem_send_module()
    module.ser = _FakeSerial(
        [
            b"Please input any key to enter X-Modem mode in 100 ms\r\n",
            b"waiting input key...\r\n",
            b"",
            b"",
            b"Send data using the xmodem protocol from your terminal\r\n",
        ]
    )
    module.args = SimpleNamespace(
        auto_reset=True,
        auto_reset_key_interval_s=0.05,
        entry_wait_read_timeout_s=0.01,
        entry_retry_reset_interval_s=0.10,
        entry_wait_timeout_s=1.0,
        port="COM5",
    )

    reset_markers: list[str] = []
    monkeypatch.setattr(module, "auto_reset_device", lambda: reset_markers.append("reset"))
    monkeypatch.setattr(module, "prime_xmodem_entry", lambda: reset_markers.append("prime"))

    ticks = (index * 0.03 for index in itertools.count())
    monkeypatch.setattr(module.time, "monotonic", lambda: next(ticks))

    module.wait_for_xmodem_ready()

    assert reset_markers[:2] == ["reset", "prime"]
    assert len(reset_markers) >= 2
    assert any(write == b"1" for write in module.ser.writes)
    assert module.ser.flush_calls >= 1


def test_wait_for_xmodem_ready_selects_xmodem_menu_entry(monkeypatch) -> None:
    module = _load_xmodem_send_module()
    module.ser = _FakeSerial(
        [
            b"waiting input key...\r\n",
            b"[1] Xmodem download and burn FW image\r\n",
            b"Send data using the xmodem protocol from your terminal\r\n",
        ]
    )
    module.args = SimpleNamespace(
        auto_reset=False,
        auto_reset_key_interval_s=0.05,
        entry_wait_read_timeout_s=0.01,
        entry_retry_reset_interval_s=0.0,
        entry_wait_timeout_s=1.0,
        port="COM5",
    )

    ticks = (index * 0.03 for index in itertools.count())
    monkeypatch.setattr(module.time, "monotonic", lambda: next(ticks))

    module.wait_for_xmodem_ready()

    assert b"1\r" in module.ser.writes


def test_wait_for_xmodem_ready_times_out_when_bootloader_never_accepts_entry(monkeypatch) -> None:
    module = _load_xmodem_send_module()
    module.ser = _FakeSerial(
        [
            b"Please input any key to enter X-Modem mode in 100 ms\r\n",
            b"waiting input key...\r\n",
            b"",
            b"",
            b"",
        ]
    )
    module.args = SimpleNamespace(
        auto_reset=True,
        auto_reset_key_interval_s=0.05,
        entry_wait_read_timeout_s=0.01,
        entry_retry_reset_interval_s=0.10,
        entry_wait_timeout_s=0.25,
        port="COM5",
    )

    monkeypatch.setattr(module, "auto_reset_device", lambda: None)
    monkeypatch.setattr(module, "prime_xmodem_entry", lambda: None)

    ticks = (index * 0.03 for index in itertools.count())
    monkeypatch.setattr(module.time, "monotonic", lambda: next(ticks))

    with pytest.raises(TimeoutError, match="COM"):
        module.wait_for_xmodem_ready()


def test_send_entry_key_reopens_serial_after_permission_error(monkeypatch) -> None:
    module = _load_xmodem_send_module()
    flaky = _FlakyWriteSerial()
    replacement = _FakeSerial([])
    module.ser = flaky

    reopen_calls: list[str] = []

    def fake_reopen_serial_port() -> None:
        reopen_calls.append("reopened")
        module.ser = replacement

    monkeypatch.setattr(module, "reopen_serial_port", fake_reopen_serial_port)

    module.send_entry_key()

    assert reopen_calls == ["reopened"]
    assert replacement.writes == [b"1"]
    assert replacement.flush_calls == 1


def test_auto_reset_device_reopens_serial_after_toggling_control_lines(monkeypatch) -> None:
    module = _load_xmodem_send_module()

    class _ResettableSerial:
        def __init__(self) -> None:
            self.dtr = None
            self.rts = None

    module.ser = _ResettableSerial()
    module.args = SimpleNamespace(
        auto_reset=True,
        auto_reset_attempts=1,
        auto_reset_pulse_s=0.0,
        auto_reset_settle_s=0.0,
    )

    reopen_calls: list[str] = []
    monkeypatch.setattr(module, "reopen_serial_port", lambda: reopen_calls.append("reopened"))
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)

    module.auto_reset_device()

    assert reopen_calls == ["reopened"]


def test_auto_reset_device_skips_local_control_lines_for_serial_urls(monkeypatch) -> None:
    module = _load_xmodem_send_module()

    class _SocketSerial:
        def __setattr__(self, name, value):
            if name in {"dtr", "rts"}:
                raise AssertionError(f"{name} should not be toggled for socket URLs")
            super().__setattr__(name, value)

    module.ser = _SocketSerial()
    module.args = SimpleNamespace(
        auto_reset=True,
        auto_reset_attempts=1,
        auto_reset_pulse_s=0.0,
        auto_reset_settle_s=0.0,
        port="socket://127.0.0.1:57005",
    )

    monkeypatch.setattr(module, "reopen_serial_port", lambda: (_ for _ in ()).throw(AssertionError("socket URL should not reopen")))

    module.auto_reset_device()


def test_dev_init_opens_socket_url_with_serial_for_url(monkeypatch) -> None:
    module = _load_xmodem_send_module()
    opened = []

    class _UrlSerial:
        def __init__(self) -> None:
            self.flush_input_calls = 0
            self.flush_output_calls = 0

        def flushInput(self):
            self.flush_input_calls += 1

        def flushOutput(self):
            self.flush_output_calls += 1

    url_serial = _UrlSerial()

    def fake_serial_for_url(port, *, baudrate, timeout):
        opened.append((port, baudrate, timeout))
        return url_serial

    monkeypatch.setattr(module.serial, "serial_for_url", fake_serial_for_url)
    monkeypatch.setattr(module, "uart_open", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("uart_open should not be used")))
    module.args = SimpleNamespace(port="socket://127.0.0.1:57005", baudrate=921600, timeout=60)

    module.dev_init()

    assert opened == [("socket://127.0.0.1:57005", 921600, 60)]
    assert module.ser is url_serial
    assert url_serial.flush_input_calls == 1
    assert url_serial.flush_output_calls == 1


def test_finalize_transfer_returns_error_immediately_on_failure(monkeypatch) -> None:
    module = _load_xmodem_send_module()
    module.ser = _FakeSerial([b"Do you want to end file transmission and reboot system? (y)\r\n"])

    sent_commands: list[str] = []
    monkeypatch.setattr(module, "send_at_command", lambda command: sent_commands.append(command))

    exit_code = module.finalize_transfer(False)

    assert exit_code == 1
    assert module.ser.readline_calls == 0
    assert sent_commands == []


def test_finalize_transfer_reboots_device_after_success(monkeypatch) -> None:
    module = _load_xmodem_send_module()
    module.ser = _FakeSerial(
        [
            b"boot noise\r\n",
            b"Do you want to end file transmission and reboot system? (y)\r\n",
        ]
    )

    sent_commands: list[str] = []
    monkeypatch.setattr(module, "send_at_command", lambda command: sent_commands.append(command))

    exit_code = module.finalize_transfer(True)

    assert exit_code == 0
    assert sent_commands == ["y"]
