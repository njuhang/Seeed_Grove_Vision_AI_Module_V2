# Himax HX6538 模型评测系统设计

**日期**: 2026-06-12
**状态**: Draft
**范围**: 第一期——模型性能评测（不含模型优化，优化另起任务）

---

## 1. 背景与目标

Seeed Grove Vision AI Module V2 基于 Himax HX6538 芯片（ARM Cortex-M55 + Ethos-U55 NPU），目前已跑通 `tflm_yolo11_od` 应用。现需建立一套系统化的模型评测体系，覆盖常见视觉和语音任务，量化各模型在目标硬件上的真实性能。

### 目标

1. 建立从 PyTorch 到板端推理的完整模型转换通路
2. 构建通用 Benchmark App，一键评测多个模型
3. 自动化 PC 端工具链：转换、烧录、结果采集、报告生成
4. 对 model_zoo 中已有模型做端到端转换验证，并与现有模型对比
5. 为后续模型优化任务提供完整的数据基础（Vela log、NPU 利用率等）

### 约束

- 推理延迟 < 1s
- 模型规模：Nano → Medium（探索平台极限）
- Flash/SRAM 资源有限，多个模型需分批烧录
- 源模型统一为 PyTorch 格式
- 模型转换环境在 WSL（litert-torch + base），烧录/串口在 Windows（串口 WSL 不可见）
- 固件与模型分离烧录：固件只烧一次，模型通过 xmodem --model 独立烧录

---

## 2. 模型集合与官方来源

所有模型均需追溯到官方 PyTorch 源。部分模型（MoveNet、MediaPipe Hands、DS-CNN 等）官方仅有 TensorFlow 实现，使用社区 PyTorch 移植版。对于 model_zoo 中已有的模型（如 YOLO11n），也需要走完整端到端转换流程，并与 model_zoo 版本对比验证。

### 2.1 目标检测（Object Detection）

| 模型 | PyTorch 来源 | 尺寸变体 | 加载方式 |
|------|-------------|----------|----------|
| **YOLOv5** | [ultralytics/yolov5](https://github.com/ultralytics/yolov5) | n/s/m | `torch.hub.load('ultralytics/yolov5', 'yolov5n')` |
| **YOLOv8** | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | n/s/m | `YOLO('yolov8n.pt')` |
| **YOLO11** | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | n/s/m | `YOLO('yolo11n.pt')` |
| **SSDLite-MobileNetV3** | [pytorch/vision](https://github.com/pytorch/vision) | Large | `torchvision.models.detection.ssdlite320_mobilenet_v3_large` |

### 2.2 图像分类（Classification）

| 模型 | PyTorch 来源 | 尺寸变体 | 加载方式 |
|------|-------------|----------|----------|
| **MobileNetV2** | [pytorch/vision](https://github.com/pytorch/vision) | — | `torchvision.models.mobilenet_v2` |
| **MobileNetV3-Small/Large** | [pytorch/vision](https://github.com/pytorch/vision) | Small, Large | `torchvision.models.mobilenet_v3_small` |
| **EfficientNet-Lite0/1** | [huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models) | 0, 1 | `timm.create_model('tf_efficientnet_lite0', pretrained=True)` |
| **SqueezeNet 1.1** | [pytorch/vision](https://github.com/pytorch/vision) | — | `torchvision.models.squeezenet1_1` |

### 2.3 语义分割（Semantic Segmentation）

| 模型 | PyTorch 来源 | 尺寸变体 | 加载方式 |
|------|-------------|----------|----------|
| **DeepLabV3-MobileNetV3** | [pytorch/vision](https://github.com/pytorch/vision) | Large | `torchvision.models.segmentation.deeplabv3_mobilenet_v3_large` |
| **LR-ASPP** | [pytorch/vision](https://github.com/pytorch/vision) | Large | `torchvision.models.segmentation.lraspp_mobilenet_v3_large` |
| **SegFormer-B0** | [NVlabs/SegFormer](https://github.com/NVlabs/SegFormer)（via [HuggingFace](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512)) | B0 | `SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")` |

### 2.4 关键点检测（Keypoint Detection）

| 模型 | PyTorch 来源 | 尺寸变体 | 加载方式 |
|------|-------------|----------|----------|
| **YOLOv8-pose** | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | n/s | `YOLO('yolov8n-pose.pt')` |
| **YOLO11-pose** | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | n/s | `YOLO('yolo11n-pose.pt')` |
| **MoveNet** | 社区移植 [fire717/movenet.pytorch](https://github.com/fire717/movenet.pytorch) | Lightning, Thunder | 非官方 PyTorch 版本 |

### 2.5 人脸检测（Face Detection）

| 模型 | PyTorch 来源 | 尺寸变体 | 加载方式 |
|------|-------------|----------|----------|
| **BlazeFace** | 社区移植 [hollance/BlazeFace-PyTorch](https://github.com/hollance/BlazeFace-PyTorch) | — | 非官方 PyTorch 版本 |
| **UltraFace** | [Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB) | RFB-320, slim-320 | 原生 PyTorch，~1MB |
| **RetinaFace-MobileNet** | [biubug6/Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface) | MobileNet-0.25 | 原生 PyTorch，~1.7MB |
| **SCRFD + MobileFaceNet（2-stage Face Recognition）** | SCRFD 检测 + MobileFaceNet embedding（优先使用 int8/static TFLite 或后续补 PyTorch 后端） | SCRFD-500M 240x320 + MobileFaceNet 112x112 | benchmark 分别计时检测与 embedding 两阶段；裁剪、对齐、相似度检索不计入模型推理性能 |

### 2.6 手势识别（Hand/Gesture）

| 模型 | PyTorch 来源 | 尺寸变体 | 加载方式 |
|------|-------------|----------|----------|
| **MediaPipe HandLandmarker** | [google-ai-edge/mediapipe](https://github.com/google-ai-edge/mediapipe)（TFLite only） | Lite, Full | 需从 TFLite 转换，或寻找社区 PyTorch 移植 |

### 2.7 语音 / 关键词识别（Audio / KWS）

| 模型 | PyTorch 来源 | 尺寸变体 | 加载方式 |
|------|-------------|----------|----------|
| **DS-CNN** | [ARM-software/ML-KWS-for-MCU](https://github.com/ARM-software/ML-KWS-for-MCU)（TF only），社区 PyTorch 版 [castorini/honk](https://github.com/castorini/honk) | Small, Medium, Large | 需用社区 PyTorch 实现 |
| **Speech Commands baseline** | [castorini/honk](https://github.com/castorini/honk) | 多种架构 | PyTorch 原生 |

### 2.8 Model Zoo 对比验证

以下 model_zoo 中已有的模型需要走完整端到端转换，并将结果与 model_zoo 版本对比：

| model_zoo 模型 | 对应 PyTorch 来源 | 对比内容 |
|----------------|-------------------|----------|
| `yolo11n_full_integer_quant_192_241219_batch_matmul_vela.tflite` | `YOLO('yolo11n.pt')` | 模型大小、延迟、NPU 利用率 |
| `yolo11n_full_integer_quant_vela_imgz_224_kris_nopost_241230.tflite` | `YOLO('yolo11n.pt')` (224 input) | 同上 |
| `model_zoo/tflm_yolov8_od/` 中的模型 | `YOLO('yolov8n.pt')` | 同上 |
| `model_zoo/tflm_mb_cls/` 中的模型 | `torchvision.models.mobilenet_v2` | 同上 |

### 2.9 分级策略

- **Tier 1（必跑）**：每个任务选 1-2 个最具代表性的原生 PyTorch 模型
- **Tier 2（扩展）**：同一模型不同尺寸（n/s/m），对比 size-accuracy-latency 曲线
- **Tier 3（探索）**：社区移植模型、Medium 级别、语音类，看边界在哪

---

## 3. 模型转换与编译流水线

### 3.1 流程

```
PyTorch 模型 (.pt / torch.hub / pip package)
        │
        ▼  [WSL] litert-torch 环境
TFLite Float32 (.tflite)
        │
        ▼  [WSL] TFLite Converter (int8 量化，含校准)
TFLite Int8 (.tflite)
        │
        ▼  [WSL] Vela Compiler (Ethos-U55 优化)
TFLite Int8 + Vela (.tflite) + Vela 日志
        │
        ▼  [共享文件系统]
Windows 侧独立烧录到 Flash
```

### 3.2 工具结构

```
tools/
├── benchmark_runner.py       # 主入口：一键编排全流程
├── model_converter/          # [WSL] 模型转换工具
│   ├── convert.py            # 一键：PyTorch → TFLite int8 → Vela
│   ├── quantize.py           # int8 量化（含校准数据集配置）
│   ├── vela_compile.py       # 调用 Vela 编译，捕获并解析日志
│   └── model_registry.py     # 读取 models.yaml
├── flash_runner/             # [Windows] 烧录 + 结果收集
│   ├── flash_firmware.py     # 烧录固件 image（仅第一次）
│   ├── flash_models.py       # 通过 xmodem --model 独立烧录模型
│   └── serial_capture.py     # pyserial 串口捕获 JSON 结果
├── report/                   # [Windows] 报告生成
│   └── generate_report.py    # 汇总 JSON + Vela info → CSV + Markdown 报告
└── configs/
    └── models.yaml           # 模型配置清单
```

### 3.3 模型配置文件（models.yaml）

```yaml
models:
  - name: yolov5n_od_192
    task: object_detection
    source:
      type: torch.hub
      repo: ultralytics/yolov5
      model: yolov5n
    input_shape: [1, 3, 192, 192]
    calibration_dataset: coco_128
    tier: 1
    flash_address: 0x200000      # Flash 偏移地址（板端映射为 0x3A200000）

  - name: yolo11n_od_192
    task: object_detection
    source:
      type: ultralytics
      model: yolo11n.pt
    input_shape: [1, 3, 192, 192]
    calibration_dataset: coco_128
    tier: 1
    flash_address: 0x400000
    model_zoo_ref: tflm_yolo11_od  # model_zoo 中已有的对应模型，用于对比验证

  - name: mobilenetv2_cls_224
    task: classification
    source:
      type: torchvision
      model: mobilenet_v2
      weights: MobileNet_V2_Weights.DEFAULT
    input_shape: [1, 3, 224, 224]
    calibration_dataset: imagenet_1k_random
    tier: 1
    flash_address: 0x600000

  - name: squeezenet11_cls_224
    task: classification
    source:
      type: torchvision
      model: squeezenet1_1
      weights: SqueezeNet1_1_Weights.DEFAULT
    input_shape: [1, 3, 224, 224]
    calibration_dataset: imagenet_1k_random
    tier: 2
    flash_address: 0x800000
```

### 3.4 Vela 日志采集

Vela 编译时捕获 stdout/stderr，解析出每个算子的调度信息：

- 哪些算子走 NPU，哪些 fallback 到 CPU
- CPU fallback 算子的名称和类型
- NPU 利用率（npu_ops / total_ops × 100%）

结构化存储为 `<model_name>.vela_info.json`，后续优化任务可直接使用。

### 3.5 校准数据集

每种任务准备一个小的校准集（几十张图），打包在仓库里或用脚本自动下载。配置在 `models.yaml` 的 `calibration_dataset` 字段中引用。

### 3.6 失败处理

某个模型转换失败不阻塞其他模型，记录错误日志，继续处理下一个。最终报告中标记失败模型和原因。

### 3.7 端到端验证

对于 `model_zoo_ref` 字段不为空的模型，转换完成后自动与 model_zoo 中的已有模型做对比：
- 模型大小差异
- Vela 编译结果差异（NPU 利用率对比）
- 板端推理延迟对比

---

## 4. 板端 Benchmark App

### 4.1 设计原则

在现有 `model_benchmark` 应用基础上扩展，引入**模型描述表**（Flash 分区表），启动时自动遍历表中所有模型并逐个推理。

### 4.2 固件与模型分离

利用现有 `xmodem_send.py` 的 `--model` 参数实现固件与模型分离烧录：

- **固件**：编译后通过 `--file` 烧录，仅在代码变更时重新烧录
- **模型**：通过 `--model` 独立烧录到指定 Flash 地址，支持一次烧多个：

```bash
# 烧录固件（仅第一次或固件变更时）
python xmodem_send.py --port=COM3 --baudrate=921600 --file=output.img

# 烧录模型（可一次烧多个）
python xmodem_send.py --port=COM3 --baudrate=921600 \
  --model="models/yolov5n_vela.tflite 0x200000 0x00000" \
  --model="models/yolo11n_vela.tflite 0x400000 0x00000" \
  --model="models/mobilenetv2_vela.tflite 0x600000 0x00000"
```

### 4.3 Flash 分区布局

```
0x00000000 ──────────── Firmware Image (前 2MB)
0x00200000 ──────────── Model Table (模型索引，独立烧录)
                [count=N (4B)]
                [entry0: name(32B), addr(4B), size(4B), input_shape(12B)]
                [entry1: ...]
                ...
0x00400000 ──────────── Model 0 (.tflite，独立烧录)
0x00600000 ──────────── Model 1 (.tflite，独立烧录)
0x00800000 ──────────── Model 2 (.tflite，独立烧录)
                ...
```

- 模型索引表和每个模型均独立烧录，地址由 PC 端 `flash_models.py` 根据 `models.yaml` 中的 `flash_address` 分配
- 地址格式：板端使用 `0x3A` 前缀映射（如 `0x00400000` → `0x3A400000`）
- 每个模型分配固定大小区域（如 2MB），避免地址冲突
- PC 端工具负责生成分区表数据并烧录到 `0x200000`

### 4.4 推理流程

```
启动
  │
  ▼
读取 Flash 分区表 (0x3A200000) → 获取 N 个模型的信息
  │
  ▼
遍历 model_table[0..N-1]:
  │
  ├── 1. 分配 Tensor Arena (mm_reserve_align)
  ├── 2. 加载模型 (GetModel)
  ├── 3. 初始化 NPU (ethosu_init)
  ├── 4. 构建 Interpreter + AllocateTensors
  ├── 5. 记录: 模型大小、Arena 使用量
  ├── 6. 填入零向量作为输入
  ├── 7. Warm-up 推理 × 2 (不计时，消除首次初始化开销)
  ├── 8. 正式推理 × NUM_BENCHMARK_RUNS 次:
  │      ├── SystemGetTick (前)
  │      ├── interpreter->Invoke()
  │      └── SystemGetTick (后) → 计算耗时
  ├── 9. 计算 avg/min/max 延迟
  ├── 10. UART 输出 JSON 结果
  └── 11. 释放 Arena，重置 NPU，准备下一个模型
```

### 4.5 测量独立性保证

**批内（同一批的多个模型之间）**：
- 每个模型跑完后完全释放 Arena，下一个模型重新分配
- NPU 状态在模型切换时重新初始化（`ethosu_init`）
- 正式计时前跑 2 次 warm-up，消除首次初始化开销

**批间（不同批次之间）**：
- 每次烧录新模型后板端硬重启，状态完全干净

### 4.6 JSON 输出格式

```json
{
  "benchmark": {
    "device": "himax_hx6538",
    "firmware": "model_benchmark_v1",
    "timestamp": 1234567890
  },
  "models": [
    {
      "name": "yolov5n_od_192",
      "task": "object_detection",
      "model_size_bytes": 4532736,
      "arena_used_bytes": 1048576,
      "latency_ms": {
        "avg": 85.3,
        "min": 82.1,
        "max": 91.7
      },
      "runs": 10,
      "status": "ok"
    }
  ]
}
```

### 4.7 配置项（common_config.h）

```c
#define MODEL_TABLE_FLASH_ADDR    0x3A200000
#define MAX_MODEL_COUNT           20
#define TENSOR_ARENA_SIZE_MAX     (2*1024*1024)
#define NUM_BENCHMARK_RUNS        10
#define WARMUP_RUNS               2
```

### 4.8 Op Resolver

在 `MicroMutableOpResolver` 中注册尽可能完整的 Op 集合，覆盖大部分常见模型：Add, Conv2D, DepthwiseConv2D, Reshape, Softmax, Transpose, MaxPool2D, AveragePool2D, FullyConnected, Concatenation, Mul, Relu, ResizeBilinear, ResizeNearestNeighbor, Split, Pad, StridedSlice, Gather, Exp, Log, ReduceMax, ReduceMin, ArgMax, EthosU 等。

---

## 5. PC 端自动化工具

### 5.1 主脚本流程

```python
def main():
    # Phase 1: [WSL] 转换所有模型
    models = load_model_registry("configs/models.yaml")
    for model in models:
        export_pytorch_model(model)     # PyTorch → 导出
        convert_to_tflite(model)        # → TFLite int8
        vela_compile(model)             # → Vela optimized + 日志采集

    # Phase 2: [WSL] 按 Flash 容量分批
    batches = compute_batches(models, flash_capacity=FLASH_SIZE)

    # Phase 3: 逐批烧录 + 采集
    all_results = []

    # [Windows] 仅第一次：烧录固件
    run_windows_cmd("python xmodem_send.py --file output.img")

    for i, batch in enumerate(batches):
        # [WSL] 生成分区表数据
        model_table = generate_model_table(batch)

        # [Windows] 烧录分区表 + 本批模型
        flash_cmd = build_model_flash_cmd(batch, model_table)
        run_windows_cmd(f"python xmodem_send.py {flash_cmd}")

        # [Windows] 触发板端重启（或等待自动重启）

        # [Windows] 串口捕获结果
        results = serial_capture(port, baudrate=921600, timeout=60)
        all_results.extend(results)

    # Phase 4: 合并板端结果 + Vela 信息，生成报告
    generate_report(all_results, vela_infos=load_vela_infos(models), output="benchmark_report")
```

### 5.2 WSL ↔ Windows 衔接

主脚本运行在 WSL 中，到烧录步骤时通过 `cmd.exe /c` 调用 Windows 侧命令：

```python
def run_windows_cmd(cmd):
    subprocess.run(["cmd.exe", "/c", cmd], check=True)
```

### 5.3 报告输出

生成两种格式，报告同时包含板端实测数据和 Vela 编译信息：

**Markdown 报告**（`benchmark_report.md`）：

```markdown
# Himax HX6538 Model Benchmark Report

**Date**: 2026-06-12
**Device**: Grove Vision AI Module V2

| Model | Task | Size (KB) | Arena (KB) | Avg (ms) | Min (ms) | Max (ms) | NPU% | CPU Fallback | Status |
|-------|------|-----------|------------|----------|----------|----------|------|-------------|--------|
| yolov5n_od_192 | OD | 4432 | 1024 | 85.3 | 82.1 | 91.7 | 100% | — | ✅ |
| deeplabv3_mbn_seg | SEG | 5210 | 1536 | 320.6 | 315.2 | 328.4 | 96.8% | ResizeNearest×2 | ⚠️ |
| yolo11n_od_192 (ours) | OD | 4530 | 1024 | 84.8 | 82.0 | 91.5 | 100% | — | ✅ |
| yolo11n_od_192 (model_zoo) | OD | 4532 | 1024 | 85.1 | 82.3 | 91.9 | 100% | — | ✅ |
```

**CSV 报告**（`benchmark_results.csv`）：同样的数据，方便后续分析绘图。

### 5.4 支持单独运行各阶段

```bash
python tools/model_converter/convert.py --model yolov5n_od     # 只转模型
python tools/flash_runner/flash_models.py --batch 1             # 只烧录某批模型
python tools/flash_runner/flash_firmware.py                     # 只烧录固件
python tools/report/generate_report.py --input results.json     # 只生成报告
```

---

## 6. 文件变更清单

| 位置 | 变更类型 | 说明 |
|------|----------|------|
| `app/scenario_app/model_benchmark/cvapp_model_benchmark.cpp` | 修改 | 增加分区表读取、多模型遍历、JSON 输出、warm-up |
| `app/scenario_app/model_benchmark/common_config.h` | 修改 | 增加分区表地址、最大模型数、warm-up 次数等配置 |
| `app/scenario_app/model_benchmark/model_benchmark.c` | 修改 | 适配新的初始化和结果输出流程 |
| `app/scenario_app/model_benchmark/model_benchmark.mk` | 修改 | 如需调整编译选项 |
| `tools/` | 新增 | PC 端自动化工具（converter、flash_runner、report） |
| `configs/models.yaml` | 新增 | 模型注册表（含官方来源、Flash 地址、model_zoo 对比引用） |

---

## 7. 不做的事（YAGNI - 本期）

- ❌ 模型精度验证（只测性能，不测 accuracy）
- ❌ 功耗测量（需要额外硬件）
- ❌ Web Dashboard（Python 报告够用）
- ❌ 动态模型下载（离线提前准备好）
- ❌ 板端真实图像采集（用零向量作为输入）
- ❌ 模型优化（算子替换、图改写等）——另起任务，但本系统通过完整记录 Vela log 和 NPU 利用率为优化任务预留数据基础

---

## 8. 扩展性预留

以下数据在本期完整采集并结构化存储，为后续优化任务提供基础：

- **Vela 编译日志**：每个算子的 NPU/CPU 调度信息，存储为 `<model_name>.vela_info.json`
- **NPU 利用率**：百分比 + CPU fallback 算子列表，纳入评测报告
- **基准性能数据**：所有模型的延迟、内存、模型大小，存储为 JSON/CSV
- **端到端验证数据**：model_zoo 对比结果，验证转换流程正确性

后续优化任务可基于这些数据：
1. 识别 NPU 利用率低的模型
2. 针对性做算子替换或图改写
3. 重新评测，对比优化前后性能
