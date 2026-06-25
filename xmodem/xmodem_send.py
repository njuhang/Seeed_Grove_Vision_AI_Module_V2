#!/usr/bin/env python
import argparse
import math
import os
import sys
import time

import serial
import xmodem

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="backslashreplace", line_buffering=True, write_through=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(errors="backslashreplace", line_buffering=True, write_through=True)

DEF_TIMEOUT = 60
DEF_BAUDRATE = 115200
DEF_PROTOCOL = "xmodem"
DEF_CHECK = "crc16"

PROTOCOL = [DEF_PROTOCOL, "xmodem1k"]
CHECK = [DEF_CHECK, "sum8"]
XMODEM_READY_TEXT = "Send data using the xmodem protocol from your terminal"
XMODEM_ENTRY_TEXTS = (
    XMODEM_READY_TEXT,
    "Please input any key to enter X-Modem mode",
    "waiting input key",
)
XMODEM_MENU_TEXTS = (
    "[1] Xmodem download and burn FW image",
    "[0] Reboot system",
)

xmodemRecType = {b"\x06": "ACK", b"\x15": "NAK"}
xmodemSndType = {0x01: "SOH", 0x02: "STX", 0x04: "EOT", 0x10: "DLE", 0x0C: "CRC", 0x18: "CAN"}
send_bin_total_packtets = 0


def format_serial_line(response):
    if isinstance(response, bytes):
        for encoding in ("utf-8", "gb18030", "latin-1"):
            try:
                return response.decode(encoding).strip()
            except UnicodeDecodeError:
                continue
        return response.decode("utf-8", errors="ignore").strip()
    return str(response).strip()


def callback(total_packets, success_count, error_count):
    if send_bin_total_packtets != 0:
        bar_total = 30
        bar_string_fmt = "\r[{}{}] {:.2%} {}/{} error: {}"
        bar_cnt = int((total_packets / send_bin_total_packtets) * bar_total)
        space_cnt = bar_total - bar_cnt

        progress = bar_string_fmt.format(
            "#" * bar_cnt,
            " " * space_cnt,
            total_packets / send_bin_total_packtets,
            total_packets,
            send_bin_total_packtets,
            error_count,
        )

        print(progress, end="")
        percent = total_packets / send_bin_total_packtets
        if percent >= 1:
            print("\n")


def uart_open(ser, com, baudrate, timeout):
    ser.port = com
    ser.timeout = timeout
    ser.baudrate = baudrate
    ser.bytesize = serial.EIGHTBITS
    ser.stopbits = serial.STOPBITS_ONE
    ser.xonxoff = 0
    ser.rtscts = 0
    ser.parity = serial.PARITY_NONE
    ser.open()
    print("Open Serial Port", ser.port)


def open_serial_endpoint(com, baudrate, timeout):
    if "://" in com:
        opened = serial.serial_for_url(com, baudrate=baudrate, timeout=timeout)
        print("Open Serial Port", com)
        return opened

    opened = serial.Serial()
    uart_open(ser=opened, com=com, baudrate=baudrate, timeout=timeout)
    return opened


def is_serial_url(port):
    return "://" in port


def dev_init():
    global ser

    try:
      ser = open_serial_endpoint(args.port, args.baudrate, args.timeout)
      ser.flushInput()
      ser.flushOutput()
    except Exception:
      print("Uart port open fail")
      sys.exit(-1)


def reopen_serial_port():
    global ser

    try:
        port = ser.port
        baudrate = ser.baudrate
        timeout = ser.timeout
    except Exception as exc:
        raise serial.SerialException(f"unable to snapshot serial settings before reopen: {exc}") from exc

    try:
        ser.close()
    except Exception:
        pass

    last_exc = None
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        try:
            reopened = open_serial_endpoint(port, baudrate, timeout)
            reopened.flushInput()
            reopened.flushOutput()
            ser = reopened
            return
        except Exception as exc:
            last_exc = exc
            time.sleep(0.1)

    raise serial.SerialException(f"failed to reopen serial port {port}: {last_exc}")


def write_serial_command(data):
    try:
        ser.write(data)
        ser.flush()
        return
    except (PermissionError, OSError, serial.SerialException):
        reopen_serial_port()
        ser.write(data)
        ser.flush()


def send_at_command(command):
    write_serial_command(bytes(command + "\r", encoding="ascii"))


def send_entry_key():
    write_serial_command(b"1")


def send_menu_choice():
    send_at_command("1")


def getc_user(size, timeout=1):
    return ser.read(size)


def putc_user(data, timeout=1):
    return ser.write(data)


def wait_for_prompt(expected_text):
    while True:
        response = ser.readline().strip()
        text = format_serial_line(response)
        print(text)
        if expected_text in text:
            return


def auto_reset_device():
    if not args.auto_reset:
        return

    if is_serial_url(getattr(args, "port", "")):
        return

    for _ in range(max(args.auto_reset_attempts, 1)):
        try:
            ser.dtr = False
            ser.rts = False
            time.sleep(args.auto_reset_pulse_s)
            ser.dtr = True
            ser.rts = True
            time.sleep(args.auto_reset_pulse_s)
            ser.dtr = False
            ser.rts = False
            time.sleep(args.auto_reset_settle_s)
            # CH343 can leave the handle in a stale state after DTR/RTS reset.
            # Reopen before sending the xmodem entry key so the key burst reaches
            # the freshly rebooted bootloader.
            reopen_serial_port()
            return
        except (AttributeError, OSError, serial.SerialException):
            time.sleep(args.auto_reset_settle_s)


def prime_xmodem_entry():
    if not args.auto_reset:
        return

    deadline = time.monotonic() + args.auto_reset_key_burst_s
    while time.monotonic() < deadline:
        send_entry_key()
        time.sleep(args.auto_reset_key_interval_s)


def wait_for_xmodem_ready():
    last_entry_key_at = time.monotonic()
    last_reset_at = last_entry_key_at
    entry_deadline = last_entry_key_at + args.entry_wait_timeout_s
    previous_timeout = ser.timeout
    ser.timeout = args.entry_wait_read_timeout_s

    try:
        while True:
            now = time.monotonic()
            if now >= entry_deadline:
                raise TimeoutError(f"timed out waiting to enter xmodem mode on {args.port}")

            response = ser.readline().strip()
            text = format_serial_line(response)
            if text:
                print(text)

            now = time.monotonic()
            if args.auto_reset and (now - last_entry_key_at) >= args.auto_reset_key_interval_s:
                send_entry_key()
                last_entry_key_at = now

            if any(prompt in text for prompt in XMODEM_ENTRY_TEXTS):
                send_entry_key()
                last_entry_key_at = now

            if any(prompt in text for prompt in XMODEM_MENU_TEXTS):
                send_menu_choice()
                last_entry_key_at = time.monotonic()

            if (
                args.auto_reset
                and args.entry_retry_reset_interval_s > 0
                and (now - last_reset_at) >= args.entry_retry_reset_interval_s
            ):
                auto_reset_device()
                prime_xmodem_entry()
                last_entry_key_at = time.monotonic()
                last_reset_at = last_entry_key_at

            if XMODEM_READY_TEXT in text:
                return
    finally:
        ser.timeout = previous_timeout


def xmodem_send_bin():
    global send_bin_total_packtets
    model_list = args.model
    image_file = args.file

    if image_file is None and model_list is None:
        print("--file and --model error parameter")
        return False

    print("Please press reset button!!")
    auto_reset_device()
    prime_xmodem_entry()
    wait_for_xmodem_ready()

    time.sleep(1)
    ser.flushInput()
    send_entry_key()

    ret = False
    wait_reboot_system = False
    packtet_size = 1024

    if args.protocol == DEF_PROTOCOL:
        packtet_size = 128

    modem = xmodem.XMODEM(getc=getc_user, putc=putc_user, mode=args.protocol)

    if image_file is not None:
        print("xmodem_sending >>", image_file)
        send_bin_total_packtets = math.ceil(os.path.getsize(image_file) / packtet_size)
        stream = open(image_file, "rb")
        ret = modem.send(stream, callback=callback)
        stream.close()

        if ret:
            print("xmodem_send bin file done!!")
            wait_reboot_system = True
        else:
            print("xmodem_send bin file FAIL!!!!")
            return ret

    if model_list is None:
        return ret

    idx = 0
    while idx < len(model_list):
        model_arg = model_list[idx].split(" ")
        if len(model_arg) < 2:
            print("--model error parameter")
            return False

        while wait_reboot_system:
            response = ser.readline().strip()
            if "Do you want to end file transmission and reboot system? (y)" in format_serial_line(response):
                break

        time.sleep(1)
        ser.flushInput()
        send_at_command("n")

        model_file = model_list[idx].split(" ")[0]
        model_file_dir = os.path.dirname(os.path.realpath(model_file))
        preamble_data = "_temp_model_" + str(idx) + "_preamble_data.bin"
        preamble_data_filepath = os.path.join(model_file_dir, preamble_data)
        print("generate {0} for {1} preamble data".format(preamble_data, model_file))

        model_position = int(model_list[idx].split(" ")[1], 16)
        model_offset = 0
        if len(model_arg) > 2:
            model_offset = int(model_list[idx].split(" ")[2], 16)

        header = [0xC0, 0x5A] + list(model_position.to_bytes(4, "little")) + list(
            model_offset.to_bytes(4, "little")
        ) + [0x5A, 0xC0] + [0xFF] * (packtet_size - 12)
        idx = idx + 1

        wb = open(preamble_data_filepath, "wb")
        wb.write(bytearray(header))
        wb.close()

        print("xmodem_sending >>", preamble_data)
        send_bin_total_packtets = math.ceil(os.path.getsize(preamble_data_filepath) / packtet_size)
        stream = open(preamble_data_filepath, "rb")
        ret = modem.send(stream, callback=callback)
        stream.close()

        if ret:
            print("xmodem_send bin file done!!")
        else:
            print("xmodem_send bin file FAIL!!!!")
            return ret

        while True:
            response = ser.readline().strip()
            if "Do you want to end file transmission and reboot system? (y)" in format_serial_line(response):
                break

        time.sleep(1)
        ser.flushInput()
        send_at_command("n")
        time.sleep(0.3)
        ser.flushInput()

        print("xmodem_sending >>", model_file)

        stream = open(model_file, "rb")
        send_bin_total_packtets = math.ceil(os.path.getsize(model_file) / packtet_size)
        ret = modem.send(stream, callback=callback)
        stream.close()

        if ret:
            print("xmodem_send bin file done!!")
        else:
            print("xmodem_send bin file FAIL!!!!")
            return ret

    return ret


def finalize_transfer(ret):
    print("xmodem_send bin file result =", ret)

    if not ret:
        return 1

    while True:
        response = ser.readline().strip()
        text = format_serial_line(response)
        print(text)
        if text == "Do you want to end file transmission and reboot system? (y)":
            print("\nFirmware upgrade completed, restart WE2 ...\n")
            send_at_command("y")
            return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port",
        required=True,
        type=str,
        help="Serial device port: COMn for windows; /dev/ttyUSBn,/dev/ttySn for unix; /dev/tty.usbserial-abcde for MacOS",
    )
    parser.add_argument("--file", type=str, help="File path for sending bin file")
    parser.add_argument(
        "--baudrate",
        default=DEF_BAUDRATE,
        type=lambda x: int(x, 0),
        help="Serial device baudrate. Default is " + str(DEF_BAUDRATE),
    )
    parser.add_argument(
        "--protocol",
        default=DEF_PROTOCOL,
        type=str,
        choices=PROTOCOL,
        help="File transfer protocol. Default is " + DEF_PROTOCOL,
    )
    parser.add_argument("--model", type=str, action="append", help='--model="bin_file flash_address_hex offset_hex"')
    parser.add_argument(
        "--timeout",
        default=DEF_TIMEOUT,
        type=lambda x: int(x, 0),
        help="Serial device timeout. Default is " + str(DEF_TIMEOUT),
    )
    parser.add_argument("--auto-reset", action="store_true", help="Toggle DTR/RTS to reboot the board into xmodem mode")
    parser.add_argument(
        "--auto-reset-pulse-ms",
        default=150,
        type=lambda x: int(x, 0),
        help="DTR/RTS pulse width in milliseconds when --auto-reset is enabled",
    )
    parser.add_argument(
        "--auto-reset-settle-ms",
        default=300,
        type=lambda x: int(x, 0),
        help="Delay after auto-reset pulse in milliseconds when --auto-reset is enabled",
    )
    parser.add_argument(
        "--auto-reset-attempts",
        default=1,
        type=lambda x: int(x, 0),
        help="Number of DTR/RTS pulse attempts when --auto-reset is enabled",
    )
    parser.add_argument(
        "--auto-reset-key-burst-ms",
        default=600,
        type=lambda x: int(x, 0),
        help="How long to keep sending the xmodem entry key after auto-reset, in milliseconds",
    )
    parser.add_argument(
        "--auto-reset-key-interval-ms",
        default=30,
        type=lambda x: int(x, 0),
        help="Delay between auto-reset key presses, in milliseconds",
    )
    parser.add_argument(
        "--entry-wait-read-timeout-ms",
        default=30,
        type=lambda x: int(x, 0),
        help="Serial readline timeout while waiting to enter xmodem mode, in milliseconds",
    )
    parser.add_argument(
        "--entry-wait-timeout-ms",
        default=30000,
        type=lambda x: int(x, 0),
        help="Total timeout while waiting to enter xmodem mode, in milliseconds",
    )
    parser.add_argument(
        "--entry-retry-reset-interval-ms",
        default=1500,
        type=lambda x: int(x, 0),
        help="While waiting for xmodem mode, re-issue auto-reset after this interval, in milliseconds",
    )

    args = parser.parse_args()
    args.auto_reset_pulse_s = max(args.auto_reset_pulse_ms, 0) / 1000.0
    args.auto_reset_settle_s = max(args.auto_reset_settle_ms, 0) / 1000.0
    args.auto_reset_key_burst_s = max(args.auto_reset_key_burst_ms, 0) / 1000.0
    args.auto_reset_key_interval_s = max(args.auto_reset_key_interval_ms, 0) / 1000.0
    args.entry_wait_read_timeout_s = max(args.entry_wait_read_timeout_ms, 1) / 1000.0
    args.entry_wait_timeout_s = max(args.entry_wait_timeout_ms, 1) / 1000.0
    args.entry_retry_reset_interval_s = max(args.entry_retry_reset_interval_ms, 0) / 1000.0

    dev_init()
    print("Device init successfully")

    ret = xmodem_send_bin()
    raise SystemExit(finalize_transfer(ret))
