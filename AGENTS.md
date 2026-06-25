# AGENTS.md

This file provides guidance to Codex and other coding agents when working in
this repository.

## Project Overview

This repository is the firmware SDK and benchmark workspace for the Seeed Grove
Vision AI Module V2, based on the Himax HX6538 / WiseEye2 chip with an ARM
Cortex-M55 and Ethos-U55 NPU. The current benchmark work focuses on converting
models to TFLite/LiteRT, compiling with Vela, flashing models separately from
firmware, and collecting board-side JSON latency / arena / status results.

## Project Rules

- WSL firmware builds must always use the repository-level `./make.sh`.
- Do not switch to manual `make`, `make clean`, or custom `GNU_TOOLPATH`
  workflows unless the user explicitly asks to debug `make.sh` itself.
- When editing sources under `EPII_CM55M_APP_S/library/inference/`, force an
  inference-library rebuild with `FORCE_INFERENCE_REBUILD=1 ./make.sh`.
- Python-based conversion, tests, and reports normally run in WSL miniforge
  environments.
- `xmodem/xmodem_send.py` is the exception when flashing a Windows COM port:
  run it with Windows Python against `COM5` or the current Windows COM port.
  Do not route Windows COM flashing through WSL unless explicitly requested.
- Generated artifacts, model weights, downloaded vendor repos, and debug
  evidence belong under `artifacts_debug/` or `artifacts/`, not at repo root.

## Known Local Environments

- WSL conda env `litert-torch`: conversion, quantization, pytest, reports.
- WSL conda env `ultralytics`: Ultralytics model tooling when needed.
- WSL conda env `vela`: standalone Vela experiments.
- Arm GNU Toolchain: `~/tool/arm-gnu-toolchain-13.2.Rel1-x86_64-arm-none-eabi/`.
- Windows serial device observed during benchmark work: `COM5`.

## Build Commands

```bash
./make.sh
```

Output image:

```text
we2_image_gen_local/output_case1_sec_wlcsp/output.img
```

Output ELF:

```text
EPII_CM55M_APP_S/obj_epii_evb_icv30_bdv10/gnu_epii_evb_WLCSP65/EPII_CM55M_gnu_epii_evb_WLCSP65_s.elf
```

When inference-library sources changed:

```bash
FORCE_INFERENCE_REBUILD=1 ./make.sh
```

## Flashing

Firmware and model flashing is done with `xmodem/xmodem_send.py`. For Windows
COM ports, run the script with Windows Python.

Firmware only:

```powershell
python xmodem\xmodem_send.py --port=COM5 --baudrate=921600 --protocol=xmodem --file=we2_image_gen_local\output_case1_sec_wlcsp\output.img
```

Firmware plus model example:

```powershell
python xmodem\xmodem_send.py --port=COM5 --baudrate=921600 --protocol=xmodem --file=we2_image_gen_local\output_case1_sec_wlcsp\output.img --model="model_zoo/tflm_yolov8_od/yolov8n_od_192_delete_transpose_0xB7B000.tflite 0xB7B000 0x00000"
```

## Selecting Application

Edit `EPII_CM55M_APP_S/makefile` and change `APP_TYPE`.

Current benchmark target:

```makefile
APP_TYPE = model_benchmark
```

Available apps include `allon_sensor_tflm`, `tflm_fd_fm`, `tflm_yolov8_od`,
`tflm_yolov8_pose`, `tflm_yolov8_gender_cls`, `tflm_peoplenet`,
`tflm_yolo11_od`, `kws_pdm_record`, `pdm_record`, `imu_read`,
`hello_world_cmsis_dsp`, `hello_world_cmsis_cv`, `ei_standalone_inferencing`,
`edge_impulse_firmware`, and `model_benchmark`.

## Key Paths

```text
configs/models.yaml                     Model manifest
tools/model_converter/                  Conversion, quantization, Vela helpers
tools/flash_runner/                     Model table, flashing, serial capture
tools/report/                           Report generation
EPII_CM55M_APP_S/app/scenario_app/      Firmware scenario apps
artifacts_debug/                        Local generated evidence and vendor assets
```

## Model Placement in Flash

- Firmware occupies the first 2 MB of flash.
- Model table is written at offset `0x200000` in the benchmark flow.
- Runtime model payloads are placed after that, typically starting at
  `0x400000`.
- Model flash addresses must be 4 KB aligned.

## Serial Communication

- Baud rate: `921600`.
- Protocol: XMODEM for firmware and model upload.
- SenseCraft Web Toolkit can be used for UART visualization when needed.
