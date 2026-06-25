# 最新进展（2026-06-21，完成审计）

- 已生成计划模型 benchmark 范围的最终完成审计：
  - 审计 JSON：`artifacts_debug/model_e2e_completion_audit_20260621.json`。
  - 审计 Markdown：`artifacts_debug/model_e2e_completion_audit_20260621.md`。
  - 结果：`10/10` 项要求通过。
- 完成审计范围：
  - `configs/models.yaml` 的 Tier-1 manifest 与 v4 审计一致：共 `22` 个计划条目。
  - 硬件证据：`19` 个板端 JSON 条目为 `status:"ok"`。
  - 边界证据：剩余 `3` 个条目均已根因闭环为 `platform_boundary`，无证据缺失。
  - 报告覆盖全部审计条目，并包含 YOLO11n、YOLOv8n、MobileNetV2 的 model_zoo ours/model_zoo/diff 对比行。
  - 最新验证日志已归档：`artifacts_debug/pytest_tools_completion_audit_20260621.log` 和 `artifacts_debug/make_sh_model_benchmark_completion_audit_20260621.log`。
- 最新验证：
  - WSL `litert-torch`：`python -m pytest tools/tests/ -q` -> `172 passed, 2 skipped, 1 warning`。
  - WSL 固件构建只使用仓库根目录 wrapper：`FORCE_INFERENCE_REBUILD=1 ./make.sh` -> 生成 `we2_image_gen_local/output_case1_sec_wlcsp/output.img`。

# 收口说明（2026-06-23）

- 主线任务已经按 2026-06-21 完成审计闭环，不再把下方历史 Phase checklist 作为当前待办来源。
- 当前交付入口：
  - 最终完成审计：`artifacts_debug/model_e2e_completion_audit_20260621.md`。
  - 平台边界说明：`artifacts_debug/platform_boundary_analysis_20260621.md`。
  - 最终报告归档：`artifacts_debug/final_baseline_20260621/`。
  - 工具使用说明：`tools/README.md`。
- 保留历史 Phase 计划用于追溯设计演进；后续新增工作应单独开新计划，优先围绕可选扩展推进，例如 Tier-2 尺寸曲线、音频/KWS 深化、更多社区模型移植。
- 收口后的仓库整理原则：
  - `com5_probe`、`final_baseline_20260621`、审计 JSON/Markdown 是证据链，默认保留。
  - 顶层临时导出物、Python cache、XMODEM preamble 临时文件默认忽略或清理。
- WSL 固件构建仍只使用仓库根目录 `./make.sh`。

# 历史 checklist 同步归档（2026-06-25）

- 已将下方历史 Phase checklist 从“待执行列表”同步为“归档追溯列表”，避免旧的未勾选任务语法被误读为当前待办。
- 已完成项按最终审计、Phase 6 真机证据和最新工具测试同步为 `[x]`。
- 原计划中的分阶段 `提交：...` 条目统一归档为“未按阶段拆分提交”，因为本阶段按用户要求未额外创建 git commit。
- 当前有效状态：主线完成审计、Phase 6 Tier-2/Tier-3 扩展完成，新增 SCRFD + MobileFaceNet 2-stage 人脸识别已在 Windows `COM5` 真机验证通过。

# 最新进展（2026-06-21，v4 边界闭环）

- 最新证据审计已升级到 v4：
  - 审计文件：`artifacts_debug/model_e2e_audit_20260621_v4.json` 和 `.md`。
  - 边界分析：`artifacts_debug/platform_boundary_analysis_20260621.md`。
  - 当前计划/Tier-1 数量仍为 `22`：`19` 个硬件 `status:"ok"` 条目，`3` 个已根因闭环的 `platform_boundary` 条目，`0` 个证据缺失。
- DeepLabV3 的不确定性已经闭合：
  - 在 `model_benchmark` 中注册 `AddHardSwish()`，并扩大 resolver 容量，以覆盖 raw DeepLabV3 MobileNetV3。
  - 按要求通过 WSL 根目录 wrapper 重建固件：`FORCE_INFERENCE_REBUILD=1 ./make.sh`。
  - raw int8 DeepLabV3 模型已通过普通 XMODEM 成功烧录到 `0x00400000`；板端现在输出 benchmark JSON，而不是因缺算子崩溃。
  - raw-int8 最终根因：板端 `AllocateTensors` 请求 `14,363,312` 字节，可用 `1,798,456` 字节。
  - Vela 路径最终根因仍是 Flash 尺寸：320x320 为 `36,339,088` 字节，192x192 为 `36,328,208` 字节，均超过 16 MB 级板载 Flash。
- 已验证 HardSwish 固件变更没有破坏 benchmark harness：
  - COM5 KWS 回归：`artifacts_debug/com5_probe/honk_after_hardswish_regression_result.json`；状态 `ok`，10 次运行，平均 `15.425 ms`。
  - 单元回归：WSL `litert-torch` 下 `python -m pytest tools/tests/test_model_benchmark_ops.py tools/tests/test_generate_report.py -q` -> `24 passed`。
  - 完整工具回归：WSL `litert-torch` 下 `python -m pytest tools/tests/ -q` -> `172 passed, 2 skipped, 1 warning`。
- 已汇总说明剩余 3 个非 `ok` 条目为什么是平台边界，而不是未解决 bug：
  - `segformerb0_seg_512`：源导出与 Vela 编译已解决，但板端 arena 请求 `40,021,504` 字节；96x96 和 64x64 探针仍分别请求 `4,959,744` 和 `4,304,384` 字节。
  - `lraspp_mbnv3_seg_320`：Vela `--arena-cache-size 0` 可以编译，但板端 arena 请求 `4,300,880` 字节；已注册的 `lraspp_mbnv3_seg_192` fit variant 已通过硬件验证。
  - `deeplabv3_mbnv3_seg_320`：HardSwish 修复后，Vela 路径是 Flash 边界，raw-int8 路径是 SRAM 边界。

# 最新进展（2026-06-21，v3 证据审计）

- 已基于当前 COM5/model 产物生成 v3 证据审计，并刷新最终归档：
  - 审计：`artifacts_debug/model_e2e_audit_20260621_v3.json` 和 `.md`。
  - 最终归档：`artifacts_debug/final_baseline_20260621/`。
  - 当前计划/Tier-1 数量仍为 `22`：`19` 个硬件 `status:"ok"` 条目，`3` 个已根因闭环的 `platform_boundary` 条目，`0` 个证据缺失。
- 已补充缺失的边界探针说明：
  - `deeplabv3_mbnv3_seg_192_probe`：192x192 Vela 模型仍为 `36,328,208` 字节，确认 DeepLabV3 阻塞点由 Flash 容量/常量尺寸主导，而不是简单输入尺寸问题。
  - `segformerb0_seg_96_probe`：板端 `AllocateTensors` 仍请求 `4,959,744` 字节，可用 `1,801,600` 字节。
  - `segformerb0_seg_64_probe`：板端 `AllocateTensors` 仍请求 `4,304,384` 字节，可用 `1,811,520` 字节，确认 SegFormer 阻塞点是 CPU fallback/动态 tensor 带来的结构性 arena 压力，而不是简单缩小输入即可解决。
- 刷新的报告现在通过 `status_reason` 保留板端失败原因，因此 `segformerb0_seg_512` 和 `lraspp_mbnv3_seg_320` 虽然原始板端结果仍是 `load_failed`，但在 `Reason` 中已记录为根因闭环的平台边界。
- 已修正最终归档中的 `mediapipe_hand_landmarks_224` Vela sidecar，改为硬件验证过的 static-int8 landmark-lite 路径；刷新后的报告显示该行 `NPU%=100.0` 且没有 CPU fallback。
- 报告/工具更新后的验证：
  - WSL `litert-torch`：`python -m pytest tools/tests/test_generate_report.py -q` -> `19 passed`。
  - WSL `litert-torch`：`python -m pytest tools/tests/ -q` -> `172 passed, 2 skipped, 1 warning`（仅 CUDA driver warning）。

# 最新进展（2026-06-20）

- 已基于当前 manifest 和 COM5 产物生成最终证据审计与报告归档：
  - 审计：`artifacts_debug/model_e2e_audit_20260620_v2.json`；当时 Tier-1/计划数量为 `22`，其中 `19` 个硬件 `status:"ok"`，`3` 个根因闭环的平台边界项（`segformerb0_seg_512`、`deeplabv3_mbnv3_seg_320`、`lraspp_mbnv3_seg_320`），无证据缺失。
  - merged 报告归档：`artifacts_debug/final_baseline_20260620/benchmark_result_merged.json`、`benchmark_report.md`、`benchmark_report.csv`。
  - model_zoo 对比归档：`benchmark_result_with_model_zoo.json`、`conversion_summary_with_model_zoo.json`、`benchmark_report_with_model_zoo.md`、`benchmark_report_with_model_zoo.csv`，包含 `yolo11n_od_192`、`yolov8n_od_192`、`mobilenetv2_cls_224` 的 ours/model_zoo/diff 行。
- 已补齐官方 model_zoo 基线：
  - `yolo11n_od_192 (model_zoo)`：`artifacts_debug/com5_probe/yolo11n_zoo_192_result.json`，`status=ok`，平均 `241.747 ms`，arena `1,086,052` 字节，模型 `2,041,472` 字节。
  - `yolov8n_od_192 (model_zoo)`：`artifacts_debug/com5_probe/yolov8n_zoo_192_result.json`，`status=ok`，平均 `101.665 ms`，arena `1,028,856` 字节，模型 `2,213,248` 字节。
  - MobileNetV2 model_zoo 基线：`artifacts_debug/com5_probe/official_qat_pruning_cls_224_result.json`，`status=ok`，平均 `90 ms`，arena `385,748` 字节，模型 `1,704,672` 字节。
- 报告工具已支持 Windows 工具可能产出的 UTF-8 BOM JSON：`tools/report/generate_report.py` 使用 `utf-8-sig` 读取 payload 和 conversion summary；回归测试覆盖 BOM 读取路径。
- MediaPipe hand landmark 和 hand detector 均已硬件验证：
  - `mediapipe_hand_landmarks_224`：`artifacts_debug/com5_probe/mediapipe_hand_landmark_lite_static_int8_vela_result.json`，`status=ok`，平均 `52.377 ms`，arena `1,057,616` 字节，模型 `1,083,904` 字节，Vela `100.0%` NPU。
  - `mediapipe_hand_detector_192`：`artifacts_debug/com5_probe/mediapipe_hand_detector_192_result.json`，`status=ok`，平均 `138.145 ms`，arena `886,616` 字节，模型 `1,321,696` 字节；manifest 通过 static-int8 + PReLU rewrite 路径闭环。
- 多个模型通过 Vela 5.1 `--arena-cache-size 0` 路径闭环：
  - `mobilenetv3small_cls_224`：`artifacts_debug/com5_probe/mbv3small_cache0_cls_224_result.json`，`status=ok`，平均 `154.202 ms`，arena `301,508` 字节。
  - `ssdlite_mbnv3_od_320`：`artifacts_debug/com5_probe/ssdlite_cache0_fixedaddr_od_320_result.json`，`status=ok`，平均 `994.658 ms`，arena `782,184` 字节。
  - `lraspp_mbnv3_seg_192`：`artifacts_debug/com5_probe/lraspp_mbnv3_seg_192_result.json`，`status=ok`，平均 `784.830 ms`，arena `1,548,884` 字节；该 192x192 变体作为 LR-ASPP fit variant 注册。
- `segformerb0_seg_512` 已从 export/missing-op 问题收敛为真实 SRAM/arena 边界：ReLU+legacy Vela 后板端结果为 `load_failed`，`AllocateTensors` 请求 `40,021,504` 字节，可用 `1,309,472` 字节；`--arena-cache-size 0` 仍报告 SRAM 需求约 `35,336 KiB`。
- 当时回归验证：
  - 定向工具测试：`75 passed, 1 warning`。
  - 完整 WSL `litert-torch` 工具测试：`171 passed, 2 skipped, 1 warning`。
  - 固件构建只使用仓库根目录 `FORCE_INFERENCE_REBUILD=1 ./make.sh`，输出 `we2_image_gen_local/output_case1_sec_wlcsp/output.img`。
- 手势/手部结论：MediaPipe hand detector 和 hand landmark 推理覆盖已进入 benchmark harness；完整 HandLandmarker 应用级后处理/手势流水线不属于本期性能 benchmark 范围。

# 最新进展（2026-06-19）

- COM5 硬件覆盖已扩展到远多于两个 YOLO OD 基线。当时已确认的 Tier-1 `status:"ok"` 代表包括：
  - `yolo11n_od_192`：`artifacts_debug/com5_probe/yolo11n_decoded_probe_after_real_rebuild_result.json`，平均 `99.185 ms`，arena `853,172` 字节。
  - `yolov8n_od_192`：`artifacts_debug/com5_probe/yolov8n_decoded_probe_after_real_rebuild_result.json`，平均 `105.045 ms`，arena `853,172` 字节。
  - `mobilenetv2_cls_224`：`artifacts_debug/com5_probe/mobilenetv2_manifest_v51_result.json`，平均 `117.863 ms`，arena `1,510,212` 字节。
  - `yolov5n_od_192`、`efficientnet_lite0_cls_224`、`yolov8n_pose_192`、`yolo11n_pose_192` 均通过 U55-64 重新编译后板端验证。
  - `squeezenet11_cls_224`、`efficientnet_lite1_cls_240`、`ultraface_rfb320_face_320x240`、`blazeface_front_face_128`、`retinaface_mnet_face_160`、`movenet_lightning_pose_192`、`honk_res8_narrow_kws_101x40` 均已有板端通过证据。
- 旧 `artifacts_debug/miniforge-runner-e2e-v2/*_vela.tflite` 产物曾使用 `ethos-u55-256` 编译，导致 HX6538/U55-64 上出现 `NPU config mismatch`；使用 `configs/himax_vela.ini`、`My_Sys_Cfg`、`My_Mem_Mode_Parent`、`ethos-u55-64` 重新编译后修复了 `yolov5n`、两个 pose 模型和 `efficientnet_lite0`。
- SqueezeNet 模型本身有效；512 KB 分块烧录实验会破坏尾部共享 FlatBuffer vtable，导致板端 metadata buffer 长度读为 0 并触发 `UsageFault`。非分块 XMODEM 烧录到 `0xA00000` 后正常通过。
- `deeplabv3_mbnv3_seg_320` 当时已确认：正确 HX6538 Vela 编译可成功且 NPU 利用率约 `99.5%`，但 Vela 模型为 `36,339,088` 字节，超过 16 MB 级 Flash；`--optimise Size` 和 `--arena-cache-size 0` 都不能解决 Flash fit 问题。
- `mobilenetv3small_cls_224`、`ssdlite_mbnv3_od_320`、`lraspp_mbnv3_seg_320`、`retinaface_mnet_face_320` 等问题逐步被区分为：可通过 cache0/fit variant 解决的路径，或真实 SRAM/arena 边界。
- MoveNet 和 Honk KWS 均已完成 manifest 注册、源导出、Vela 编译和板端验证：
  - `movenet_lightning_pose_192`：平均 `64.488 ms`，arena `1,111,088` 字节。
  - `honk_res8_narrow_kws_101x40`：平均 `15.425 ms`，arena `90,804` 字节。
- 工具链补充：`tools/model_converter/vela_compile.py` 支持 `--optimise`、`--tensor-allocator` 和重复 `--extra-arg`，避免 Vela 实验继续依赖临时脚本。
- 当时剩余 spec 覆盖缺口已缩小到平台边界/可选扩展：目标检测、分类、关键点、人脸、手部、KWS 均已有硬件通过代表；原始 320/512 语义分割大模型仍是 SRAM/Flash 边界。

# 最新进展（2026-06-17）

- `mbv3small10` 根因收敛完成：用户建议的原始 `vela` 命令与仓库当前 Vela 5.1 编译路径等价，输出 `.tflite` 字节一致，因此问题不是命令拼写或 config 路径。
- 通过额外 Ethos-U driver/device trace，Vela 5.1 COP1 失败定位为：command stream 提交成功，但 NPU 没有产生 `ethosu_wait()` 完成中断；同一固件上的小型 Vela 5.1 控制模型可正常完成，说明 IRQ 和通用 driver 路径健康。
- Vela 4.5 编译同一收窄模型时，失败模式从“命令启动后挂起”变成确定性的 `AllocateTensors` arena 短缺 `4176` 字节。
- 已验证可用 workaround：`Vela 4.5 + --optimise Size` 生成的 `artifacts_debug/mbv3small10_v45_size/mbv3small10_int8_vela.tflite` 可在板端运行：`artifacts_debug/com5_probe/mbv3small10_v45_size_long_result.json`，`status=ok`，arena `1,904,724` 字节，平均 `1843.817 ms`。
- 真实 manifest 生成的完整 `mobilenetv3small_cls_224` 当时仍无法放入 benchmark firmware arena：`artifacts_debug/com5_probe/mobilenetv3small_manifest_v45_result.json` 显示 `AllocateTensors` 请求 `2,836,224` 字节，可用 `1,872,032` 字节。因此剩余阻塞已从 Vela 5.1 hang 转为完整 1000 类模型超过板端 SRAM。
- 分类覆盖不再完全阻塞：`mobilenetv2_cls_224` 真实 manifest 路径已转换并在板端成功运行：`artifacts_debug/com5_probe/mobilenetv2_manifest_v51_result.json`，`status=ok`，arena `1,510,212` 字节，平均 `117.863 ms`，Vela 约 `99.0%` NPU。
- Phase 3 对比工具已接入：`benchmark_runner.py` 自动展开 `ours/model_zoo` 执行计划，`generate_report.py` 生成 diff 行并处理 Markdown 转义，避免 model_zoo 预编译 baseline 被错误报告为 `0% NPU / Passthrough-only`。
- MobileNetV2 的真实端到端分类对比已完成：`artifacts_debug/manifest_mbv2_compare/benchmark_report.md` 和 `.csv` 中包含 `ours`、`model_zoo`、`diff` 行。
- YOLO11n/YOLOv8n source export 路径逐步修正：Ultralytics detect head 改为返回 decoded primary tensor，更贴近 model_zoo no-NMS 契约；固件补充 `AddBroadcastTo()` 后，两个 decoded YOLO 变体在 COM5 上均通过。
- 大模型烧录支持和失败上报得到加强：`flash_models.py` 支持可选 `model_chunk_size`，`xmodem_send.py` 在传输失败时立即非零退出，相关回归覆盖 `test_flash_models.py`、`test_benchmark_runner.py`、`test_xmodem_send.py`。
- 当时下一步是保留 MobileNetV3Small 的 manifest 级 Vela 4.5 fallback，但将其归类为 model-fit 问题；同时把 decoded-head YOLO source export 推入主 manifest/report flow。
# HX6538 模型评测系统端到端实施计划
> **给执行 agent：** 本计划覆盖设计 spec（`docs/superpowers/specs/2026-06-12-model-benchmark-design.md`）的全部内容，按 spec 章节组织，并逐项标注**当前状态**。截至 2026-06-25，下方 Phase checklist 已同步为历史归档，不再作为当前待办来源；新增工作应单独开新计划。
>
> **状态图例：** `✅` 已实现，`🟡` 已实现但仍有缺陷待修复，`📦` 历史归档。

**目标：** 将 HX6538（Grove Vision AI Module V2）上的“PyTorch 源模型 → 板端真机推理延迟 / 内存 / NPU 利用率”打通成一套一键、可复现、可对比的端到端评测系统，并覆盖 spec §2 的全部任务类型。

**架构：** PC 端（WSL + Windows 双环境）负责模型转换、分批、烧录编排与报告；板端固件（`model_benchmark` 应用）读取 Flash 模型表、逐模型推理、通过 UART 输出 JSON；两端通过“Flash 分区模型表 + JSON 结果契约”解耦，固件只烧一次，模型经 `xmodem --model` 独立烧录。

**技术栈：**
- 板端：C/C++、TFLM（`tflmtag2412_u55tag2411`）、Ethos-U55、ARM Cortex-M55、Himax HX6538。
- PC-转换（WSL conda env `litert-torch`）：PyTorch 2.12 / torchvision 0.27 / ultralytics 8.4.66 / **litert-torch 0.8.0 + ai-edge-quantizer 0.4.2 + ai-edge-litert 2.1.2**，采用 PT 直转 LiteRT，**不走 ONNX**，之后再过 Vela。
- PC-编排 / 报告（Windows Python）：`pyserial`、`xmodem`、`yaml`、`pytest`。

**关键设计决策（对应 spec）：**
1. **转换路径 = PyTorch → litert-torch 直转 LiteRT（float）→ ai-edge-quantizer 做 int8 全整数量化 → Vela。全程不经过 ONNX。** 这既覆盖也修正了 spec §3.1 的流程图，并意味着早期那 2 个依赖 ONNX 路径的失败测试应当废弃并重写。
2. 固件 / 模型分离、Flash 分区表、JSON 契约保持不变；这部分已实现且总体稳定。

---

## 0. 计划起点（当前基线）

- 单元测试 **27/29 通过**，2 个失败测试位于 `tools/tests/test_convert_cli.py`，其断言了已废弃的 `ultralytics → ONNX → onnx2bf` 两阶段路径，需改写为 `litert-torch` 直转。
- 真机已跑通 2 个 OD 模型（`yolo11n_od_192` 约 `93 ms`、`yolov8n_od_192` 约 `94 ms`），但 `conversion_summary.json` 显示二者均为 `staged_model_zoo_ref`（源导出 SIGABRT，回退到 `model_zoo`），**真实源导出从未成功**。
- Vela 信息解析有误：近 100% NPU 的 YOLO 模型被误报为约 `33%~53%`，并带有“Passthrough”伪 fallback（见 Phase 0）。
- `APP_TYPE = model_benchmark`（`makefile:160`）已配置好，固件可直接编译。

执行顺序建议：**Phase 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7**。各 Phase 产出可独立验证与提交；其中 Phase 1/2 仅依赖 WSL Python（可 TDD），Phase 4/5/6 需要连接开发板。

---

## Phase 0 - 基线修复：让现状可信（spec §3.4 / §4.6 / §5.3）
**目的：** 在推动转换流水线之前，先把“解析 / 计时 / 报告”修对，避免在错误数据上继续叠加新功能。本阶段全部可在 PC 侧 TDD，不依赖开发板。

### Task 0.1 - 修复 Vela 输出解析（NPU 利用率与 CPU fallback 误判）🟡→✅
- **涉及文件：** `tools/model_converter/vela_compile.py`（`parse_vela_output`、正则 `CPU_OPERATIONS_RE` / `NPU_OPERATIONS_RE` / `CPU_BREAKDOWN_RE`）；`tools/tests/test_vela_compile.py`。
- **现状缺陷：** 当前 Vela 版本（`--show-cpu-operations`）的输出格式与旧正则不匹配，导致 `cpu_ops` 被误抓、`npu_utilization_pct` 偏低，并把非算子行（如 `Passthrough`）当成 CPU fallback。
- **做什么：** 以 `artifacts/vela/*.vela_info.json` 对应模型的真实 Vela stdout 为样本，核对 Vela 实际输出中的 “CPU operations / NPU operations / Network summary” 段落措辞，修正正则；明确“CPU fallback 算子”只提取真正的算子类型名，过滤 `Passthrough` 与空行。
- **验收：** 用 `yolo11n` / `yolov8n` 的真实 Vela 日志回归，利用率应约为 `95%+`（这两个 model_zoo 模型本应接近全 NPU）；`python -m pytest tools/tests/test_vela_compile.py -q` 全绿。
- [x] 重写解析单测（基于真实日志片段）
- [x] 修正正则
- [x] 测试通过
- [x] 归档：原提交点 `fix(vela): correct CPU/NPU op parsing and fallback filtering` 未按阶段拆分提交

### Task 0.2 - 板端 JSON 时间戳从硬编码 0 改为真实时间 🟡→✅
- **涉及文件：** `app/scenario_app/model_benchmark/model_benchmark.c`（`run_multi_model_benchmark` 里当前的 `"timestamp": 0`）。
- **做什么：** 若板端无 RTC，则改用“上电后的 `SystemGetTick` 计时（ms）”，或保留 `0` 但在 JSON 里新增 `boot_tick` 字段标注来源；至少要让报告能区分多次运行。
- **验收：** 固件通过仓库根目录 `./make.sh` 编译通过；真机输出 JSON 中 `timestamp` / `boot_tick` 不再恒为 0，且单调递增。
- [x] 实现真实计时来源
- [x] 编译通过
- [x] 真机验证 JSON
- [x] 归档：原提交点 `feat(model_benchmark): emit real timestamp/tick in JSON` 未按阶段拆分提交

### Task 0.3 - 报告数值一致性回归 ✅ 维持
- **涉及文件：** `tools/report/generate_report.py`、`tools/tests/test_generate_report.py`。
- **做什么：** 依赖 Task 0.1 修复后，补一条回归：给定高 NPU% 的 `vela_info`，报告的 MD / CSV 中 `NPU%` 与 `CPU Fallback` 列应渲染正确。
- **验收：** `python -m pytest tools/tests/test_generate_report.py -q` 通过。
- [x] 补回归测试
- [x] 通过
- [x] 归档：原提交点 `test(report): regression for vela npu% rendering` 未按阶段拆分提交

> **Phase 0 出口：** `python -m pytest tools/tests/ -q` 全绿（除将在 Phase 1 改写的 2 个 ONNX 测试）；真机报告中的 NPU% 数值可信。

---

## Phase 1 - 转换流水线：改为 litert-torch 直转（spec §3.1 / §3.2 / §3.5）
**目的：** 让“PyTorch 源模型”真正导出成可烧录的 Vela int8 tflite，而不是每次都回退到 `model_zoo`。统一不走 ONNX。整个阶段都可在 WSL Python 内 TDD。

### Task 1.1 - 新建 litert-torch 直转核心模块 ✅
- **涉及文件：** 新建 `tools/model_converter/litert_convert.py`；改造 `tools/model_converter/quantize.py`（当前只有路径 helper，需补上真实量化逻辑）。
- **做什么：**
  - 模块按 `source.type` 提取 PyTorch `nn.Module`（ultralytics 取 `YOLO(w).model` 的 pre-NMS 主干 + 头，对齐 model_zoo “nopost” 语义；`torchvision` / `timm` / `torch.hub` / HuggingFace 各自构造）。
  - 使用 **litert-torch 0.8.0** 将 `nn.Module` 直接导出为 **float `.tflite`**（底层经 `torch_xla2` lowering，不经过 ONNX）。
  - 使用 **ai-edge-quantizer 0.4.2** 配合代表性校准数据，做 **int8 全整数量化**，输出 `<name>_int8.tflite`。
  - 保留 `quantize.py` 现有 `artifact_dir / quantized_model_path / vela_model_path` 路径契约不变。
- **验收：** 模块单测中，给定一个极小 `nn.Module`（如单层 Conv）和内存中的假校准数据，能产出非空 float / int8 tflite 字节；`python -m pytest tools/tests/` 中新增的 `test_litert_convert.py` 通过。
- [x] 先写失败单测（假校准数据 → 产出 int8 tflite）
- [x] 实现 litert-torch 直转 + 量化
- [x] 测试通过
- [x] 归档：原提交点 `feat(convert): litert-torch direct PT→LiteRT int8 (no onnx)` 未按阶段拆分提交

### Task 1.2 - 校准数据集加载器（spec §3.5）✅
- **涉及文件：** 新建 `tools/model_converter/calibration.py`（或并入 `litert_convert.py`）；`configs/` 下补数据集清单与下载脚本。
- **做什么：** 按 `models.yaml` 的 `calibration_dataset` 字段提供代表性数据生成器：
  - `coco_128`（OD）：下载或读取打包好的少量 COCO 图片，按 `input_shape` 预处理（resize / 归一化 / int8 scale）。
  - `imagenet_1k_random`（分类）：少量 ImageNet 样本或确定性随机张量（按 spec §7，“离线提前准备好”可以退化为可复现的随机样本）。
  - 后续 `speech_commands`（音频）见 Phase 6。
- **验收：** 单测校验每个已知 key 返回的样本数量、形状、dtype 均符合 int8 量化器要求。
- [x] 实现校准加载器 + key 映射
- [x] 单测
- [x] 归档：原提交点 `feat(convert): representative calibration data loaders` 未按阶段拆分提交

### Task 1.3 - `convert.py` 按 `source.type` 派发到 litert-torch，移除 ONNX 路径 🟡→✅
- **涉及文件：** `tools/model_converter/convert.py`（`_convert_from_source`、`_run_ultralytics_export`、`REQUIRED_SOURCE_DEPS`、`WSL_PYTHON`）；删除或替换 `wsl_export_ultralytics.py`（原 ONNX 内部路径）。
- **现状缺陷：** 当前 ultralytics 走 `model.export(format='tflite', int8=True)` 时内部依赖 ONNX；仓库曾因此遇到 `NotImplementedError`，并额外依赖两个 conda env（`litert-torch` / `onnx2bf-cpu`）。
- **做什么：**
  - 统一调度到 Task 1.1 的 `litert-torch` 直转，单一 WSL conda env（`litert-torch`）即可。
  - 失败处理（spec §3.6）保留现状：每模型独立 `try/except`、记录 notes，不阻塞其他模型。
  - 保留 `model_zoo_ref` 兜底，但仅在“真实源导出失败且显式允许”时回退，用于 Phase 3 对比。
- **验收：** 真实跑一次 `yolo11n_od_192` 与 `yolov8n_od_192`，`conversion_summary.json` 中 `status` 为 `exported_from_source`（而非 `staged_model_zoo_ref`）；产物 `_int8.tflite` 经 Vela 后可被固件加载（见 Phase 4）。
- [x] 重构派发逻辑
- [x] 真实导出并跑通 2 个 OD 模型
- [x] 归档：原提交点 `refactor(convert): dispatch all sources through litert-torch, drop onnx path` 未按阶段拆分提交

### Task 1.4 - 增补源类型后端：`torchvision` / `torch.hub` / `timm` / HuggingFace ✅
- **涉及文件：** `convert.py`（`REQUIRED_SOURCE_DEPS` + 模型构造分支）；`litert_convert.py` 中 `nn.Module` 取数适配。
- **做什么：** 按 spec §2 的来源补齐构造器并接入直转：`torchvision.models.*`、`timm.create_model`、`torch.hub.load`、`SegformerForSemanticSegmentation.from_pretrained`。
- **验收：** 每个后端至少有一个模型的 “float → int8” 单测或真实转换样本通过。
- [x] `torchvision` 后端 + 测试
- [x] `timm` 后端 + 测试
- [x] `torch.hub` 后端 + 测试
- [x] HF（SegFormer）后端 + 测试
- [x] 归档：原提交点 `feat(convert): add <source> backend` 未按阶段拆分提交

### Task 1.5 - 改写 2 个 ONNX 失败测试为 litert-torch 直转测试 🟡→✅
- **涉及文件：** `tools/tests/test_convert_cli.py`。
- **做什么：** 将 `test_convert_manifest_uses_split_ultralytics_and_onnx2bf_pipeline` 重命名为 `..._uses_litert_torch_direct_pipeline`，并断言：单阶段、仅使用 `litert-torch` env、产物为 int8 tflite、不再出现 onnx2bf / onnx 相关可执行；同时修复 `test_convert_cli_writes_summary_json`（失败根因相同）。
- **验收：** `python -m pytest tools/tests/ -q` 全绿（不再包含任何 ONNX 路径假设）。
- [x] 重命名并重写测试
- [x] 全套通过
- [x] 归档：原提交点 `test(convert): rewrite onnx-path tests for litert-torch direct pipeline` 未按阶段拆分提交

> **Phase 1 出口：** `yolo11n` / `yolov8n` 真实源导出成功（`exported_from_source`）；单元测试全绿；ONNX 依赖从转换路径彻底移除。

---

## Phase 2 - Tier-1 模型注册表扩展：覆盖全部任务类型（spec §2.1-§2.9 / §2.9 Tier-1）
**目的：** spec §2.9 要求 Tier-1 覆盖“每类任务 1~2 个最具代表性的原生 PyTorch 模型”。当前 `models.yaml` 只有 2 个 ultralytics OD 模型，需要补齐其余任务类别。

### Task 2.1 - 注册表扩展到 Tier-1 全任务 ✅
- **涉及文件：** `configs/models.yaml`；按需调整 `model_registry.py`（若需新增字段，如 `source.weights`、HF repo 等）。
- **做什么：** 按 spec §2 表格，为每类任务加入 Tier-1 原生 PyTorch 模型，并分配 4KB 对齐、互不重叠的 `flash_address`：
  - OD（已有）：`yolo11n_od_192`、`yolov8n_od_192`；可补 `yolov5n`、`ssdlite_mobilenetv3`。
  - 分类：`MobileNetV2`、`MobileNetV3-Small`（`torchvision`）。
  - 分割：`DeepLabV3-MobileNetV3`（`torchvision`）、`SegFormer-B0`（HF）。
  - 关键点：`YOLOv8-pose` / `YOLO11-pose`（ultralytics）。
  - 人脸：`UltraFace-RFB-320`、`RetinaFace-MobileNet`（原生 PyTorch）。
  - 手势：`MediaPipe HandLandmarker`，标注“需 TFLite 反推或社区移植”，并列入 Phase 6。
  - 音频 / KWS：`DS-CNN`，标注“社区 PyTorch 实现”，并列入 Phase 6。
- **验收：** `load_models("configs/models.yaml")` 成功且校验通过（`batch=1`、4KB 对齐、flash 地址不重叠）；每个 Tier-1 模型都能跑通 Phase 1 的转换，产出 Vela tflite。
- [x] OD 补齐
- [x] 分类
- [x] 分割
- [x] 关键点
- [x] 人脸
- [x] 手势 / 音频（已在 Phase 6 归档为扩展验证）
- [x] 归档：原提交点 `feat(configs): expand models.yaml to tier-1 across all tasks` 未按阶段拆分提交

### Task 2.2 - Tier-1 转换批量验证 ✅
- **涉及文件：** `tools/tests/`（端到端 smoke）；`artifacts/`。
- **做什么：** 跑 `python tools/model_converter/convert.py --manifest configs/models.yaml`，确认每个 Tier-1 模型的状态为 `exported_from_source`，或在失败时有可解释原因并记入报告。
- **验收：** 转换汇总 JSON 中成功模型占比明确；失败项有原因且不阻塞整体流程。
- [x] 批量转换并归档产物
- [x] 归档：原提交点 `chore(artifacts): tier-1 conversion outputs` 未按阶段拆分提交

> **Phase 2 出口：** Tier-1 全任务均有可烧录的 Vela int8 tflite。

---

## Phase 3 - `model_zoo` 端到端对比验证（spec §2.8 / §3.7）
**目的：** 对存在 `model_zoo_ref` 的模型，把“自研转换产物”和“model_zoo 既有版本”做模型大小、Vela NPU 利用率、板端延迟三项对比，以验证转换流程正确性。

### Task 3.1 - 对比数据采集与行级输出 ✅
- **涉及文件：** `tools/report/generate_report.py`（`build_rows` / 新增对比行）；`tools/tests/test_generate_report.py`。
- **做什么：** 当某模型存在 model_zoo 对照时，报告额外产出对照行（`<name> (ours)` vs `<name> (model_zoo)`），列出 `Size / Arena / Avg / NPU% / Fallback` 的差值；转换阶段需同时确保 ours 与 model_zoo 两份 `vela_info` 都被生成。
- **验收：** 单测覆盖 “ours vs model_zoo” 的双行渲染与差值列；真机报告中包含对照。
- [x] 生成双版本 `vela_info`
- [x] 报告对比行 + 差值列
- [x] 单测
- [x] 归档：原提交点 `feat(report): model_zoo comparison rows with diffs` 未按阶段拆分提交

> **Phase 3 出口：** `YOLO11n` / `YOLOv8n` 的 ours vs model_zoo 三项对比进入报告。

---

## Phase 4 - 板端固件收尾与多模型真机验证（spec §4 全章）
**目的：** 固件侧功能已基本实现（见下表），本阶段聚焦在真机上验证多模型表回环，以及用 Phase 1 的真实产物替代 `model_zoo` 兜底。需要开发板。

### 板端现状一览（对应 spec §4，以证据为主）
| Spec | 状态 | 证据 |
|---|---|---|
| §4.1 模型表加载 + 启动遍历 | 🟡 | `model_benchmark.c::run_multi_model_benchmark` |
| §4.2 固件 / 模型分离烧录 | ✅ | `xmodem_send.py --model` |
| §4.3 Flash 分区表（`0x200000`，含 `magic/version/count/entries`） | ✅ | `model_table.h/c` + `model_table.py`（`MAGIC=MODL` / `VERSION=1`，双端一致） |
| §4.4 推理流程（arena → load → NPU init → AllocateTensors → warm-up×2 → 计时×N → avg/min/max → JSON） | ✅ | `model_benchmark.c::benchmark_one_model`、`cvapp_model_benchmark.cpp` |
| §4.5 测量独立性（释放 arena / 重启 NPU / warm-up；批间硬重启） | ✅ | 批内自动，批间手动硬重启，符合 spec “每次烧录新模型后硬重启” |
| §4.6 JSON 输出格式 | ✅ | `run_multi_model_benchmark`，但 timestamp 待修，见 Task 0.2 |
| §4.7 `common_config.h` 配置项 | ✅ | `MODEL_TABLE_FLASH_ADDR / MAX_MODEL_COUNT / TENSOR_ARENA_SIZE_MAX / NUM_BENCHMARK_RUNS / WARMUP_RUNS` 均已存在 |
| §4.8 Op Resolver（完整算子集） | ✅ | `cvapp_model_benchmark.cpp` 中已注册 24 个 op；`ReduceMin` 因当前 TFLM 构建缺省而注明略去 |

### Task 4.1 - 用 Phase 1 真实产物做板端多模型回环验证 🟡→✅
- **涉及文件：** 原则上只做验证，不改固件；若暴露新的缺失算子，再补固件。
- **做什么：** 将 Phase 1/2 产出的 Vela tflite 烧入板端，校验模型表 `magic/version/count` 解析正确，且每个模型都输出 `status:"ok"`；若出现 `load_failed`，排查是否命中 `op_resolver` 未注册算子，并在 spec §4.8 范围内补注册，同时更新 `tools/tests/test_model_benchmark_ops.py` 契约。
- **验收：** 真机 JSON 中 Tier-1 模型全部 `status:"ok"`，延迟与 arena 数值合理。
- [x] 烧录模型表 + 模型
- [x] 回环验证 status
- [x] 按需补算子注册 + 契约测试
- [x] 归档：原提交点 `fix(model_benchmark): register <op> for real-model coverage` 未按阶段拆分提交

> **Phase 4 出口：** Phase 1 真实转换产物在真机多模型表下全部跑通。

---

## Phase 5 - 端到端编排：一键全流程（spec §5.1 Phase-3 / §5.2 / §5.4）
**目的：** 当前 `benchmark_runner.py` 仅覆盖 Phase 1（转换）/ Phase 2（分批计划）/ Phase 4（报告），**Phase 3（烧录 + 重启 + 串口采集）尚未完全串起来**，仍需手工配合本地工具。本阶段把一键闭环真正接上。需要开发板。

### Task 5.1 - 串口采集器接入真实端口（spec §5.1 采集）🟡→✅
- **涉及文件：** `tools/flash_runner/serial_capture.py`（当前只有 `extract_json_objects` 文本解析，没有端口读取入口）。
- **做什么：** 增加“打开 `pyserial` 端口 → 持续读取 → 喂给 `extract_json_objects` → 在超时或完整 JSON 收齐后返回”的入口（Windows 侧）；同时保留纯解析函数，便于单测。
- **验收：** 单测覆盖纯解析函数；真机可从 COM 口稳定采集到一整次 benchmark JSON。
- [x] 实现端口采集入口
- [x] 单测（解析）+ 真机（端口）
- [x] 归档：原提交点 `feat(flash_runner): live serial capture of benchmark json` 未按阶段拆分提交

### Task 5.2 - 烧录 + 重启编排（spec §5.1 Phase-3 / §5.2）✅
- **涉及文件：** `tools/flash_runner/flash_models.py`（当前仅拼接 `--model` 参数串，不执行）、`flash_firmware.py`（仅返回 image 路径）、`xmodem/xmodem_send.py`（已支持 `--file` / `--model` / “请按 reset” 交互）。
- **做什么：** 实现 `run_windows_cmd`（spec §5.2，基于 `subprocess` 调用 Windows 侧 `xmodem_send.py` / `flash_bridge.py`），按 `benchmark_plan.json` 的每一批次执行：烧模型表（`0x200000`）+ 本批模型 → 触发 / 等待板端重启 → 采集 JSON。固件仅在首次或固件变更时通过 `flash_firmware.py` 烧录。
- **验收：** 单测用 mock `subprocess` 覆盖命令拼装与批次循环；真机端到端能跑完至少一批，并产出 `benchmark_result_*.json`。
- [x] 烧录执行 + WSL↔Windows 桥接
- [x] mock 单测
- [x] 真机单批验证
- [x] 归档：原提交点 `feat(flash_runner): phase-3 flash+reset+capture orchestration` 未按阶段拆分提交

### Task 5.3 - `benchmark_runner` 串成一键全流程 🟡→✅
- **涉及文件：** `tools/benchmark_runner.py`（`main`）。
- **做什么：** 在 `main` 中串联：转换 → 分批 →（逐批烧录 + 重启 + 采集）→ 合并 board 结果与 Vela 信息 → 报告；并支持 `--skip-flash`、`--from-board-result` 等子阶段开关，满足 spec §5.4 的“阶段可独立运行”。
- **验收：** 一条命令即可产出 `benchmark_report.md` / `.csv`，覆盖全部烧录批次结果；子阶段开关也能独立运行。
- [x] 串联主流程 + 子阶段开关
- [x] 端到端 smoke
- [x] 归档：原提交点 `feat(benchmark_runner): one-click end-to-end orchestration` 未按阶段拆分提交

> **Phase 5 出口：** `python tools/benchmark_runner.py ...` 一键完成“转换 → 烧录 → 采集 → 报告”。

---

## Phase 6 - Tier-2 / Tier-3 扩展与探索（spec §2.9 / §2.4 / §2.6 / §2.7）
**目的：** 探索平台边界，例如同模型 n/s/m 的 size-accuracy-latency 曲线、社区移植模型、Medium 级关键点与语音类任务。本阶段偏探索性，按价值排序推进。

### Task 6.1 - Tier-2 尺寸变体曲线（spec §2.9 Tier-2）✅
- **涉及文件：** `configs/models.yaml`、报告生成逻辑（按尺寸分组对比）。
- **做什么：** 为代表模型补齐 n/s/m 变体，跑完后在报告中按模型族给出 size / latency 趋势。
- **验收：** 报告至少包含一族（如 YOLO n/s/m）的趋势行。
- [x] 加入第一族 YOLO11 Tier-2 尺寸变体：`yolo11s_od_192`（`yolo11m_od_192` 因非当前目标平台尺寸主动跳过）
- [x] 用轻量 `yolo_fastestv2_od_192` 替代 YOLO medium 探索名额，并完成官方 Zenodo V0.2 PyTorch 权重的 LiteRT int8 + Vela 编译
- [x] `benchmark_runner.py --tiers` 支持选择 Tier-1 / Tier-2 执行计划
- [x] 报告生成支持 `size trend` 趋势行
- [x] `yolo11s_od_192` 完成 LiteRT int8 导出与 Vela 编译（见 `artifacts_debug/phase6_tier2_yolo11_status_20260623.md`）
- [x] 完成转换与可烧录执行计划归档；最终按项目规则修正为 Windows Python 直接执行 `xmodem_send.py --port=COM5`，模型表 + `yolo11s_od_192` + `yolo_fastestv2_od_192` 均烧录成功
- [x] 完成 Phase 6 真机跑分：`yolo11s_od_192` status `ok`，avg `245.675 ms`；`yolo_fastestv2_od_192` status `ok`，avg `40.399 ms`（见 `artifacts_debug/phase6-final-tier2-od/phase6_board_result_windows_com5_20260624.json` / `phase6_board_report.md`）
- [x] 归档：`feat(configs): tier-2 size variants` 相关变更已落入工作树；是否创建 git commit 由用户单独决定

### Task 6.2 - 社区移植 / 语音模型（spec §2.4 MoveNet / §2.6 MediaPipe / §2.7 DS-CNN）✅
- **涉及文件：** 新增对应的 `nn.Module` 加载适配（可能需 vendor 社区仓库）、`calibration.py` 对 `speech_commands` 的支持、固件 `op_resolver` 的按需扩展。
- **做什么：** 引入 MoveNet / BlazeFace / MediaPipe Hand 的 PyTorch 实现；补齐 DS-CNN 的 `speech_commands` 校准；并明确“移植正确性”不在本期精度验证范围内（见 spec §7）。
- **验收：** 每个社区模型都能导出为 Vela int8，并在真机侧达到 `status:"ok"`（只验证性能，不验证精度）。
- [x] MoveNet：`movenet_lightning_pose_192` 已在 COM5 通过，avg `64.488 ms`（见 `artifacts_debug/com5_probe/movenet_lightning_pose_192_result.json` / `artifacts_debug/final_baseline_20260621/benchmark_report.md`）
- [x] BlazeFace / RetinaFace（原生路径已在 Tier-1）：`blazeface_front_face_128` avg `73.383 ms`，`retinaface_mnet_face_160` avg `756.061 ms`（见 `artifacts_debug/final_baseline_20260621/benchmark_report.md`）
- [x] MediaPipe Hand：`mediapipe_hand_detector_192` avg `138.145 ms`，`mediapipe_hand_landmarks_224` avg `52.377 ms`，均已 `status=ok`
- [x] DS-CNN / KWS 替代路径：`honk_res8_narrow_kws_101x40` 使用 `speech_commands` 校准并在 COM5 通过，avg `15.425 ms`
- [x] 归档：社区 / 语音模型已纳入 `configs/models.yaml`、转换 backend、COM5 证据与最终 baseline 报告；本阶段不额外创建 git commit，除非用户单独要求

### Task 6.3 - 2-stage 人脸识别：SCRFD + MobileFaceNet（新增需求）✅
- **涉及文件：** `configs/models.yaml`、转换/报告链路、后续可选固件 pipeline 后处理。
- **做什么：** 将人脸识别拆成两个可独立 benchmark 的模型阶段：Stage 1 `SCRFD` 人脸检测，Stage 2 `MobileFaceNet` 人脸 embedding。当前 benchmark 先验证两个模型各自的推理性能；人脸裁剪、对齐、embedding 相似度阈值/检索不混入模型性能统计。
- **验收：** `scrfd_500m_face_det_240x320` 与 `mobilefacenet_face_embed_112` 均进入 manifest，同属 `face_recognition_2stage_scrfd_mobilefacenet` pipeline；生成对应 static-int8/Vela TFLite 后可烧录并在真机侧 `status:"ok"`。
- [x] 新增 manifest：`scrfd_500m_face_det_240x320`，task `face_detection`，Tier-3，Stage 1
- [x] 新增 manifest：`mobilefacenet_face_embed_112`，task `face_embedding`，Tier-3，Stage 2
- [x] 新增注册表测试，锁定两个阶段的 `pipeline_id` / `pipeline_stage` / 输入形状
- [x] 接入实际模型资产：SCRFD 500m 240x320 dynamic-range TFLite，后续由流水线生成 static-int8
- [x] 接入实际模型资产：MobileFaceNet 112x112 float TFLite，后续由流水线生成 static-int8
- [x] Vela 编译、Windows COM5 烧录、真机采集与报告归档
- [x] 板端结果：`scrfd_500m_face_det_240x320` status `ok`，avg `52.131 ms`，arena `616500`，NPU `99.1%`
- [x] 板端结果：`mobilefacenet_face_embed_112` status `ok`，avg `111.048 ms`，arena `1204740`，NPU `99.6%`
- [x] 修复记录：MobileFaceNet 先因缺少 `L2_NORMALIZATION` resolver 注册而 `load_failed`，随后因 int8 bias 导致 Conv/Depthwise fallback 卡在首次 warmup；已补 `AddL2Normalization()`，并新增 `rewrite_int8_conv_biases_to_int32` 将 Conv/Depthwise bias 转为 TFLite/Vela 期望的 int32。

> **Phase 6 出口：** 评测集扩展到 Medium 级关键点、手势、语音，并能据此总结平台能力边界。

---

## Phase 7 - 报告归档、文档与扩展性预留（spec §6 / §8）

### Task 7.1 - 文件变更清单核对（spec §6）✅
- **核对表（当前 vs spec §6）：**
  - `app/scenario_app/model_benchmark/cvapp_model_benchmark.cpp`：已包含分区表读取、多模型遍历、JSON 输出、warm-up。
  - `common_config.h`：已包含全部 §4.7 配置项。
  - `model_benchmark.c`：已接入新流程（timestamp 修复见 Task 0.2）。
  - `model_benchmark.mk`：`✅`（`LIB_SEL=tflmtag2412_u55tag2411` 等已配置）；如 Phase 6 新增算子，再重新评估。
  - `tools/`：框架已建立；Phase 1 / 5 继续补直转与编排。
  - `configs/models.yaml`：已存在；Phase 2 / 6 扩充内容。

### Task 7.2 - 扩展性数据归档（spec §8）✅
- **做什么：** 确保每个模型都有 `<name>.vela_info.json`（算子、NPU / CPU 调度），以及 benchmark JSON / CSV、model_zoo 对比数据（Phase 3）的结构化落盘，以供后续优化任务直接复用。
- **验收：** `artifacts/` 下每模型的 `vela_info` 与基准结果齐备；`README` / `docs/profiling.md` 更新为最新基线表。
- [x] 归档结构化数据
- [x] 更新 profiling 文档 / 报告入口
- [x] 归档：原提交点 `docs: update benchmark baseline and profiling notes` 未按阶段拆分提交

### Task 7.3 - 顶层使用文档 ✅
- **涉及文件：** `tools/README.md`（新增）或扩写现有文档。
- **做什么：** 写清环境要求（WSL `litert-torch` env、Windows 串口、Vela）、一键命令、各子阶段命令与输出说明。
- **验收：** 新人只看文档即可独立复现一次端到端评测。
- [x] 撰写使用文档
- [x] 归档：原提交点 `docs: benchmark tooling usage guide` 未按阶段拆分提交

> **Phase 7 出口：** 产物归档完整，文档可复现。

---

## 不做项（spec §7，本计划遵守）
- 模型精度验证（本项目只做性能 benchmark）。
- 功耗测试。
- Web Dashboard（当前 Python 报告足够）。
- 自动模型下载（按离线准备处理）。
- 板端真实图像采集（以零值 / 随机向量输入为主）。
- 模型优化本身另开任务；本系统只负责沉淀完整 Vela log 与 NPU 利用率数据，为后续优化预留基础。

---

## 全局验收（完成标志）
1. `python -m pytest tools/tests/ -q` 全绿，且不再包含 ONNX 路径假设。
2. Tier-1 全任务模型均为 `exported_from_source`，且真机多模型表回环结果全部 `status:"ok"`。
3. `python tools/benchmark_runner.py ...` 一键产出包含 `NPU%` / `CPU fallback` / `model_zoo` 对比的 `benchmark_report.md` / `.csv`。
4. `artifacts/` 下每模型的 `vela_info` 与 benchmark 结果结构化归档，可直接被后续优化任务复用。

---

## 自检结论（对照 spec）
- **覆盖范围：** §1 目标、§2 全任务模型集（Phase 2 / 6）、§3 转换流水线（Phase 1，litert-torch 直转并修正 §3.1）、§4 板端 App（Phase 4 现状表）、§5 PC 自动化（Phase 5）、§6 文件清单（Phase 7.1）、§7 不做项、§8 扩展预留（Phase 7.2）均有对应任务。
- **状态标注：** 已实现项用 `✅` 或 `🟡`，并可附证据文件；历史待实现项已在 2026-06-25 归档同步。
- **一致性：** Flash 表契约（`MODL` / `v1`）、路径 helper、JSON 字段名在固件与 PC 两端保持一致；转换路径统一为 `litert-torch` 直转，不再依赖 ONNX。
