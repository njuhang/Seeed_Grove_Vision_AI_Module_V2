# NPU 算子优化记录

本文记录每一种算子/fallback 优化规则，或一次有明确结论的调试进展。只有
PC 侧 Vela 指标和板端 benchmark 证据都写清楚，后续才容易复现实验。

## 记录模板

### YYYY-MM-DD - 简短标题

- Op/Fallback Type（Op/Fallback 类型）：
- Models（涉及模型）：
- Failure Symptom（失败现象）：
- Root Cause Hypothesis（根因假设）：
- Implementation Method（实现方法）：
- Debug Process（关键 debug 步骤）：
- Vela Before/After（Vela 前后指标）：
- Board Verification（板端验证结果）：
- Remaining Risk（剩余风险）：

## 规则与进展

### 2026-06-25 - 基线安全 TFLite rewrite 规则

- Op/Fallback 类型：FLOAT16 DEQUANTIZE 常量、float PReLU island、非法 INT8 Conv/DepthwiseConv bias、缺失的 `serving_default` signature。
- 涉及模型：首轮自动优化目标包括 `segformerb0_seg_512`、`retinaface_mnet_face_160`、`yolov8n_pose_192`、`mediapipe_hand_landmarks_224`。
- 失败现象：Vela 将部分子图留在 CPU 侧，降低 NPU 覆盖率。
- 根因假设：部分 fallback 来自 TFLite 图表示细节，可以在不改变模型意图的前提下规范化。
- 实现方法：在 `tools/model_converter/tflite_rewrite.py` 中注册安全 rewrite 规则，由 `tools/model_converter/npu_optimizer.py` 运行候选、重编 Vela 并按 NPU%、CPU ops、fallback 类型和模型大小选择最优结果。
- 关键 debug 步骤：先保存 baseline Vela sidecar，再应用候选 rewrite，重跑 Vela，对比 `cpu_ops`、`npu_ops`、`npu_utilization_pct` 和 `cpu_fallback_ops`，最后保留 baseline 与 optimized 两套产物。
- Vela 前后指标：每次运行写入对应模型目录下的 `optimizer_trace.json`，例如 `artifacts_debug/npu_opt_full_validation_20260625/<model>/optimizer_trace.json`。
- 板端验证结果：本轮完整报告见 `artifacts_debug/npu_opt_full_validation_20260625/report.md`。
- 剩余风险：PC 侧 Vela 改善不等于板端成功，Flash/SRAM/resolver 都可能成为新的边界；因此 baseline 产物必须保留。

### 2026-06-25 - SegFormer B0 DepthwiseConv bias rewrite 移除一种 fallback

- Op/Fallback 类型：`DepthwiseConv2DBias` fallback。
- 涉及模型：`segformerb0_seg_512`。
- 失败现象：baseline Vela 只有 `86.5%` NPU，CPU fallback 包含 `Transpose, Conv2DBias, Mean, FullyConnected, BatchMatMul, DepthwiseConv2DBias, ResizeBilinear`。
- 根因假设：部分 depthwise conv bias tensor 类型/量化表示不符合 Vela 期望，导致该段不能委派给 Ethos-U。
- 实现方法：启用 `all_safe_rewrites`，其中 INT8 Conv/DepthwiseConv bias 会转换为 INT32 bias，随后重新运行 Vela。
- 关键 debug 步骤：检查 `segformerb0_seg_512/optimizer_trace.json`，确认候选 rewrite 后 `DepthwiseConv2DBias` 从 fallback 类型列表中消失，并由 optimizer 选中 optimized Vela 产物。
- Vela 前后指标：`npu_utilization_pct 86.5 -> 92.1`，CPU ops `154 -> 87`，fallback 类型从 `Transpose,Conv2DBias,Mean,FullyConnected,BatchMatMul,DepthwiseConv2DBias,ResizeBilinear` 变为 `Transpose,Conv2DBias,Mean,FullyConnected,BatchMatMul,ResizeBilinear`。
- 板端验证结果：optimized 产物已烧录并运行，但 `AllocateTensors` 失败；串口日志显示 `Requested: 36249600, available 1446964, missing: 34802636`。报告中状态为 `npu_opt_board_failed`，说明本次是 PC 侧有效优化、板端 SRAM 边界失败，不能计为最终优化成功。
- 剩余风险：需要后续继续做 activation/SRAM 压缩或模型结构级优化；仅 TFLite rewrite 不足以让该模型板端通过。

### 2026-06-25 - RetinaFace 安全 rewrite 无 PC 侧收益但板端可运行

- Op/Fallback 类型：`Passthrough` fallback。
- 涉及模型：`retinaface_mnet_face_160`。
- 失败现象：Vela NPU 覆盖率只有 `39.4%`，但 fallback 类型只有 `Passthrough`。
- 根因假设：低 NPU% 主要来自图中 Vela 无法切分/委派的 passthrough 子图，而不是当前安全 rewrite 能修复的 bias 或 float island 问题。
- 实现方法：运行 `all_safe_rewrites` 候选并重编 Vela；optimizer 因无 PC 侧改善而保留 baseline 产物。
- 关键 debug 步骤：检查 `retinaface_mnet_face_160/optimizer_trace.json`，确认 baseline 与 candidate 均为 `39.4%` NPU、CPU ops `114`、NPU ops `74`。
- Vela 前后指标：`39.4% -> 39.4%`，fallback 仍为 `Passthrough`，无有效优化。
- 板端验证结果：板端 `status=ok`，平均延迟约 `756.066 ms`，说明低 NPU 覆盖率下仍可运行，但性能较弱。
- 剩余风险：需要针对 passthrough 切分原因做更细的 op/shape 级分析；当前第一版 rewrite 无法改善。

### 2026-06-25 - Pose 模型 STABLEHLO_PAD 板端 resolver 边界

- Op/Fallback 类型：`STABLEHLO_PAD`。
- 涉及模型：`yolov8n_pose_192`、`yolo11n_pose_192`。
- 失败现象：Vela 编译完成，但板端 `AllocateTensors` 阶段打印 `Didn't find op for builtin opcode 'STABLEHLO_PAD'`，随后进入 `BusFault_Handler`，没有输出 benchmark JSON。
- 根因假设：导出的 TFLite 模型中残留 StableHLO PAD builtin，当前 benchmark firmware 的 TFLM resolver 没有注册该 op；这不是 Vela rewrite 已覆盖的问题。
- 实现方法：本轮未实现 op 支持，只在 runner 中增加 `continue_on_flash_error` 容错，使该类无 JSON 崩溃记录为 `benchmark_timeout` 并继续后续 batch。
- 关键 debug 步骤：查看 `serial_benchmark_capture_batch11.log` 与 `serial_benchmark_capture_batch12.log`，确认失败点均在 `STABLEHLO_PAD` resolver 缺失。
- Vela 前后指标：`yolov8n_pose_192` optimizer 触发但无改善，`94.0% -> 94.0%`，fallback 为 `Passthrough`；`yolo11n_pose_192` 为 `95.5%`。
- 板端验证结果：两个 pose batch 均未得到 JSON，报告状态为 `benchmark_timeout`。`squeezenet11_cls_224` 与 `yolo11n_pose_192` 同 batch，因前一个模型 BusFault 未被执行，也记录为 batch timeout。
- 剩余风险：需要优先尝试导出阶段消除 StableHLO op，或确认当前 TFLM 版本是否可注册对应 kernel；否则 pose 模型无法完成板端 benchmark。

### 2026-06-25 - MediaPipe hand landmarks LOGISTIC resolver 边界

- Op/Fallback 类型：`LOGISTIC`。
- 涉及模型：`mediapipe_hand_landmarks_224`。
- 失败现象：Vela 编译得到 `0.0%` NPU，板端 `AllocateTensors` 阶段打印 `Didn't find op for builtin opcode 'LOGISTIC'`，随后 BusFault。
- 根因假设：当前 benchmark firmware resolver 未注册 `LOGISTIC`，同时该模型基本无法被 Ethos-U 委派，属于 CPU/resolver 路径边界。
- 实现方法：本轮未实现 `LOGISTIC` 支持；runner 容错后将该模型记录为 `benchmark_timeout`，避免阻塞整批验证。
- 关键 debug 步骤：查看 `serial_benchmark_capture_batch20.log`，确认失败点为 `LOGISTIC` resolver 缺失。
- Vela 前后指标：optimizer 触发但无改善，`0.0% -> 0.0%`，CPU ops `63`，NPU ops `0`，fallback 为 `Passthrough`。
- 板端验证结果：未输出 benchmark JSON，报告状态为 `benchmark_timeout`。
- 剩余风险：即使补齐 `LOGISTIC`，该模型仍可能主要跑 CPU，需要重新评估是否适合作为 NPU 优化目标。

### 2026-06-25 - 两阶段人脸识别链路板端通过

- Op/Fallback 类型：`Passthrough` fallback，2-stage face pipeline 模型组合。
- 涉及模型：`scrfd_500m_face_det_240x320`、`mobilefacenet_face_embed_112`。
- 失败现象：新增两阶段人脸识别需求后，需要确认 detection 与 embedding 两个模型能否在同一批次烧录并板端运行。
- 根因假设：两个模型大小较小，Vela NPU 覆盖率较高，主要风险在 batch 打包地址和 runtime arena。
- 实现方法：使用现有转换/Vela/benchmark runner，无需新增 rewrite 规则。
- 关键 debug 步骤：将 `scrfd` 与 `mobilefacenet` 和 `yolo_fastestv2` 放在 batch16，确认模型表地址分别为 `0x400000`、`0x45a000`、`0x4fd000`，并检查板端 JSON。
- Vela 前后指标：`scrfd_500m_face_det_240x320` NPU `99.1%`，`mobilefacenet_face_embed_112` NPU `99.6%`，fallback 均为 `Passthrough`。
- 板端验证结果：两者均 `status=ok`；SCRFD 平均延迟约 `52.130 ms`，MobileFaceNet 平均延迟约 `111.047 ms`。
- 剩余风险：当前 benchmark 只验证模型加载和固定输入推理耗时，尚未验证两阶段业务级后处理、face crop 对齐和 embedding 数值质量。
