"""
Test Conv+SiLU with FP8 quantization and SiLU LUT.
- Conv: input FP8, weights FP8, accumulation in float (decode then conv; equivalent to INT32 acc with scale).
- SiLU: piecewise linear / LUT output FP8.
Measures quantization error vs reference FP32 ONNX run.
"""

import os
import numpy as np

try:
    import onnx
    import onnxruntime as ort
except ImportError:
    raise SystemExit("Install onnx and onnxruntime: pip install onnx onnxruntime")

from fp8_e4m3 import fp32_to_fp8_e4m3, fp8_e4m3_to_fp32
from silu_lut import silu_fp8_lut, silu_fp8_lut_to_fp32


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ONNX_PATH = os.path.join(SCRIPT_DIR, "conv_silu.onnx")


def load_onnx_conv_weights(onnx_path: str):
    """Load Conv weight and bias from ONNX initializers."""
    model = onnx.load(onnx_path)
    w = None
    b = None
    for init in model.graph.initializer:
        if init.name and "weight" in init.name.lower():
            w = np.frombuffer(init.raw_data, dtype=np.float32).reshape(
                tuple(init.dims)
            ).copy()
        if init.name and "bias" in init.name.lower():
            b = np.frombuffer(init.raw_data, dtype=np.float32).reshape(
                tuple(init.dims)
            ).copy()
    if w is None:
        for init in model.graph.initializer:
            if len(init.dims) == 4:
                w = np.frombuffer(init.raw_data, dtype=np.float32).reshape(
                    tuple(init.dims)
                ).copy()
                break
    return w, b


def run_reference_fp32(onnx_path: str, input_fp32: np.ndarray) -> np.ndarray:
    """Run ONNX model in FP32 (reference)."""
    sess = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"],
    )
    out, = sess.run(
        None,
        {sess.get_inputs()[0].name: input_fp32.astype(np.float32)},
    )
    return out


def conv2d_nchw(x: np.ndarray, w: np.ndarray, b: np.ndarray | None) -> np.ndarray:
    """NCHW conv2d with same padding (pad = kernel_size // 2)."""
    n, c, h, w_in = x.shape
    oc, ic, kh, kw = w.shape
    ph, pw = kh // 2, kw // 2
    x_pad = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)), mode="constant", constant_values=0)
    out = np.zeros((n, oc, h, w_in), dtype=np.float32)
    for ni in range(n):
        for o in range(oc):
            for hi in range(h):
                for wi in range(w_in):
                    s = 0.0
                    for i in range(ic):
                        for ki in range(kh):
                            for kj in range(kw):
                                s += x_pad[ni, i, hi + ki, wi + kj] * w[o, i, ki, kj]
                    out[ni, o, hi, wi] = s
            if b is not None:
                out[ni, o, :, :] += b[o]
    return out


def run_quantized_conv_silu_fp8(
    input_fp32: np.ndarray,
    w_fp32: np.ndarray,
    b: np.ndarray | None,
) -> np.ndarray:
    """
    Quantized path: input FP8, W FP8, conv in float (decode FP8 -> float, conv),
    then quantize conv output to FP8, SiLU via LUT (FP8 -> FP8), decode to float.
    """
    x_fp8 = fp32_to_fp8_e4m3(input_fp32)
    w_fp8 = fp32_to_fp8_e4m3(w_fp32)
    x_fp32 = fp8_e4m3_to_fp32(x_fp8)
    w_fp32_q = fp8_e4m3_to_fp32(w_fp8)
    conv_out_fp32 = conv2d_nchw(x_fp32, w_fp32_q, b)
    conv_out_fp8 = fp32_to_fp8_e4m3(conv_out_fp32)
    silu_out_fp32 = silu_fp8_lut_to_fp32(conv_out_fp8)
    return silu_out_fp32


def main():
    if not os.path.isfile(ONNX_PATH):
        print("Run create_conv_silu_onnx.py first to generate", ONNX_PATH)
        return

    w_fp32, b = load_onnx_conv_weights(ONNX_PATH)
    if b is None:
        b = np.zeros(w_fp32.shape[0], dtype=np.float32)

    np.random.seed(42)
    # Input shape must match the ONNX model (1, 2, 8, 8)
    input_fp32 = np.random.randn(1, 2, 8, 8).astype(np.float32) * 0.5

    ref = run_reference_fp32(ONNX_PATH, input_fp32)
    quant = run_quantized_conv_silu_fp8(input_fp32, w_fp32, b)

    abs_err = np.abs(quant - ref)
    mse = np.mean((quant - ref) ** 2)
    max_abs = np.max(abs_err)
    mean_abs = np.mean(abs_err)
    # relative error where ref != 0
    mask = np.abs(ref) > 1e-7
    rel_err = np.abs(quant - ref) / (np.abs(ref) + 1e-12)
    mean_rel = np.mean(rel_err[mask]) if np.any(mask) else float("nan")
    # cosine similarity (1 = identical direction)
    r, q = ref.ravel(), quant.ravel()
    nr, nq = np.linalg.norm(r), np.linalg.norm(q)
    cos_sim = (np.dot(r, q) / (nr * nq + 1e-12)) if (nr > 1e-12 and nq > 1e-12) else float("nan")

    print("=== Conv+SiLU FP8 quantization test ===")
    print("Conv: input FP8, W FP8; SiLU: LUT output FP8")
    print()
    print("Quantization error (vs FP32 reference):")
    print("  MSE:             ", mse)
    print("  Max abs error:   ", max_abs)
    print("  Mean abs error:  ", mean_abs)
    print("  Mean rel error:  ", mean_rel)
    print("  Cosine similarity:", cos_sim)
    print()
    print("Reference output stats: min={:.6f} max={:.6f} mean={:.6f}".format(
        ref.min(), ref.max(), ref.mean()
    ))
    print("Quantized output stats: min={:.6f} max={:.6f} mean={:.6f}".format(
        quant.min(), quant.max(), quant.mean()
    ))


if __name__ == "__main__":
    main()
