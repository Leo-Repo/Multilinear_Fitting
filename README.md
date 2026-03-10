# SiLU FP8 (E4M3) Quantization via Lookup Table

## 任务说明 (Task)

- **目标**：用查表法实现 SiLU 函数的 FP8 (e4m3) 量化。
- **测试设定**：使用 Conv+SiLU 的 ONNX 模型，约定如下：
  - Conv 输入：FP8
  - Conv 权重：FP8
  - 累加器：INT32（本实现中在 float 域等价实现：FP8 解码后做卷积再量化）
  - SiLU：分段线性/查表输出为 FP8
- **评估**：相对 FP32 参考的量化误差（MSE、最大/平均绝对误差、平均相对误差、余弦相似度等）。

## 实现概要

### FP8 E4M3 格式

- **规格**：1 位符号 + 4 位指数 + 3 位尾数，bias = 7（OFP8 / NVIDIA E4M3 风格）。
- **范围**：正常数约 ±2⁻⁶ ~ ±448；次正规数可表示到 ±2⁻⁹；无无穷，NaN 为 `0x7F`/`0xFF`。
- **模块**：`fp8_e4m3.py` — `fp32_to_fp8_e4m3` / `fp8_e4m3_to_fp32`（标量及数组接口）。

### SiLU 查表 (LUT)

- **定义**：SiLU(x) = x · sigmoid(x) = x / (1 + exp(-x))。
- **LUT**：对 256 个 FP8 输入码字，先解码为 float，计算 SiLU，再量化回 FP8，得到 256 项查找表。
- **模块**：`silu_lut.py` — `build_silu_fp8_lut()`、`silu_fp8_lut()`（FP8 入 → FP8 出）、`silu_fp8_lut_to_fp32()`（便于与 FP32 参考比较）。

### 测试流程 (Conv + SiLU)

1. **参考**：ONNX 模型 FP32 推理（Conv → SiLU）得到参考输出。
2. **量化路径**：输入与权重量化为 FP8 → 解码后做卷积（等价 INT32 累加）→ 卷积结果再量化为 FP8 → SiLU 查表得到 FP8 输出 → 解码为 float 用于误差计算。
3. **指标**：MSE、最大绝对误差、平均绝对误差、平均相对误差（仅对参考非零点）、余弦相似度（1 表示方向一致）。

## 项目结构

```
Multilinear_Fitting/
├── README.md                 # 本说明与报告
├── task.txt                  # 原始任务描述
├── requirements.txt          # Python 依赖
├── fp8_e4m3.py               # FP8 E4M3 编解码
├── silu_lut.py               # SiLU FP8 查表实现
├── create_conv_silu_onnx.py  # 生成 Conv+SiLU ONNX（需 torch）
├── test_conv_silu_fp8.py     # Conv+SiLU FP8 量化误差测试
└── conv_silu.onnx            # 测试用 ONNX（需先生成）
```

## 环境与依赖

```bash
# 核心依赖（推理与测试）
pip install -r requirements.txt   # numpy, onnx, onnxruntime

# 生成 ONNX 时额外需要
# pip install torch
```

推荐使用 Conda 环境（如 `llama`）并保证已安装 `onnx`、`onnxruntime`；若需自行导出 ONNX，再安装 `torch`。

## 使用步骤

### 1. 生成 Conv+SiLU ONNX（若尚未生成）

```bash
conda activate llama   # 或你的环境
python create_conv_silu_onnx.py
```

会生成 `conv_silu.onnx`（默认：输入 `(1, 2, 8, 8)`，单层 Conv2d + SiLU）。

### 2. 运行 FP8 量化误差测试

```bash
conda activate llama
python test_conv_silu_fp8.py
```

## 测试结果（报告用）

在默认随机种子与上述 Conv+SiLU 设置下，FP8 量化相对 FP32 参考的典型误差如下：

| 指标 | 数值 |
|------|------|
| **MSE** | 0.00065 |
| **最大绝对误差** | 0.116 |
| **平均绝对误差** | 0.017 |
| **平均相对误差** | ~32.9% |
| **余弦相似度 (cosine similarity)** | ≈ 0.998（越接近 1 表示与参考方向越一致） |

- **参考输出 (FP32)**：min ≈ -0.24，max ≈ 0.65，mean ≈ 0.018。
- **量化输出 (FP8)**：min ≈ -0.20，max ≈ 0.56，mean ≈ 0.013。

结论：在 Conv 输入/权重 FP8、累加等价为 INT32、SiLU 查表输出 FP8 的设定下，误差量级符合 8-bit 量化预期，可用于算法验证与报告。

## 许可与引用

本项目为 FP8 SiLU 查表量化与 Conv+SiLU 测试的实现与报告用代码，按项目需求使用即可。
