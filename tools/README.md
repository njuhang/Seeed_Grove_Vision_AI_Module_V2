# Benchmark Tooling

This directory contains the PC-side tooling for the HX6538 model benchmark flow.

## WSL environments

The current workflow uses three WSL `miniforge` environments:

- `litert-torch`
  Runs PyTorch -> LiteRT export, int8 quantization, the Python test suite, and
  the top-level `benchmark_runner.py`.
- `ultralytics`
  Optional sandbox for upstream model inspection or standalone Ultralytics
  experiments.
- `vela`
  Runs Arm Ethos-U Vela 5.1.0.

The tools auto-discover Vela from `~/miniforge3/envs/vela/bin/vela`. Override
with `VELA_EXECUTABLE=/path/to/vela` if needed.

## Common commands

Run tool tests, including the real conversion smoke checks:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate litert-torch
cd /mnt/d/code/Seeed_Grove_Vision_AI_Module_V2
RUN_REAL_CONVERSION_SMOKE=1 python -m pytest tools/tests -q
```

Convert one model from source to int8 TFLite without Vela:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate litert-torch
cd /mnt/d/code/Seeed_Grove_Vision_AI_Module_V2
python tools/model_converter/convert.py \
  --manifest configs/models.yaml \
  --output-dir artifacts_debug/yolo11n-probe \
  --vela-dir artifacts_debug/yolo11n-probe/vela \
  --skip-vela \
  --isolate-source-exports \
  --model yolo11n_od_192
```

Run the PC-side batch flow (convert + Vela + flash plan generation):

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate litert-torch
cd /mnt/d/code/Seeed_Grove_Vision_AI_Module_V2
python tools/benchmark_runner.py \
  --run-convert \
  --isolate-source-exports \
  --output-dir artifacts_debug/miniforge-runner-e2e \
  --report-prefix artifacts_debug/miniforge-runner-e2e/report
```

Run the Phase 6 Tier-2 size-variant exploration path:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate litert-torch
cd /mnt/d/code/Seeed_Grove_Vision_AI_Module_V2
python tools/benchmark_runner.py \
  --tiers 1,2 \
  --run-convert \
  --isolate-source-exports \
  --output-dir artifacts_debug/phase6-tier2-yolo11 \
  --report-prefix artifacts_debug/phase6-tier2-yolo11/report
```

Generate a report from an existing board JSON payload:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate litert-torch
cd /mnt/d/code/Seeed_Grove_Vision_AI_Module_V2
python tools/report/generate_report.py \
  --input artifacts/benchmark_result_latest.json \
  --manifest configs/models.yaml \
  --vela-dir artifacts_debug/miniforge-runner-e2e/vela \
  --conversion-summary artifacts_debug/miniforge-runner-e2e/conversion_summary.json \
  --output-prefix artifacts_debug/miniforge-runner-e2e/report
```

## Current status

As of 2026-06-21, the Tier-1 benchmark scope is closed by audit:

- Completion audit: `artifacts_debug/model_e2e_completion_audit_20260621.md`.
- Final baseline archive: `artifacts_debug/final_baseline_20260621/`.
- Planned Tier-1 entries: 22.
- Hardware `status:"ok"` entries: 19.
- Root-caused platform boundaries: 3.
- Model-zoo comparison rows are present for YOLO11n, YOLOv8n, and MobileNetV2.

Latest recorded verification:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate litert-torch
cd /mnt/d/code/Seeed_Grove_Vision_AI_Module_V2
python -m pytest tools/tests/ -q
```

Expected recorded result: `172 passed, 2 skipped, 1 warning`.

Firmware build verification must use the repository-level wrapper:

```bash
cd /mnt/d/code/Seeed_Grove_Vision_AI_Module_V2
FORCE_INFERENCE_REBUILD=1 ./make.sh
```

Expected recorded result: `we2_image_gen_local/output_case1_sec_wlcsp/output.img`.

The remaining non-`ok` planned entries are documented platform limits, not open
conversion bugs:

- `segformerb0_seg_512`: SRAM/arena pressure from CPU fallback and dynamic tensors.
- `lraspp_mbnv3_seg_320`: SRAM/arena pressure; the 192x192 fit variant passes.
- `deeplabv3_mbnv3_seg_320`: Flash-size boundary on the Vela path and SRAM boundary
  on the raw-int8 path.

As of 2026-06-24, Phase 6 development has started with the first Tier-2
target-platform-relevant OD expansion:

- `configs/models.yaml` includes `yolo11s_od_192` as a Tier-2 entry alongside
  the existing Tier-1 `yolo11n_od_192`.
- `yolo11m_od_192` is intentionally skipped; the medium size is outside the
  current target-platform scope.
- `yolo_fastestv2_od_192` is added as a lightweight YOLO-FastestV2 candidate.
  Its `yolo_fastestv2` source backend loads the official Zenodo V0.2 PyTorch
  repo/weights and exports raw detector heads through LiteRT + Vela.
- `tools/benchmark_runner.py --tiers` controls which manifest tiers enter the
  flash/execution plan. The default remains `--tiers 1`.
- `tools/report/generate_report.py` appends a `size trend` row when two or
  more YOLO n/s/m/x variants in the same family have board `status:"ok"`.

Phase 6 final conversion artifacts are under
`artifacts_debug/phase6-final-tier2-od/`:

- `phase6_summary.md` summarizes conversion size, Vela size, NPU utilization,
  CPU fallback, batch, flash address, and board-validation status.
- `benchmark_plan.json` is ready to flash: model table at `0x200000`,
  `yolo11s_od_192` at `0x400000`, and `yolo_fastestv2_od_192` at `0xc0d000`.
- `serial_access_diagnostic_20260624.md` records the current COM5/WSL visibility
  diagnostic, the socket bridge attempt, and the final direct Windows COM5
  flashing path.
- `hardware_flash_attempts_20260624.md` records the failed socket bridge reset
  attempt and the final successful Windows `COM5` `xmodem_send.py` run.
- `phase6_board_result_windows_com5_20260624.json` and `phase6_board_report.md`
  are the board evidence: `yolo11s_od_192` passed at `245.675 ms` average and
  `yolo_fastestv2_od_192` passed at `40.399 ms` average.
- `phase6_completion_audit_20260624.md` closes Phase 6 by tying Task 6.1 and
  Task 6.2 requirements to current conversion, report, and COM5 board evidence.
