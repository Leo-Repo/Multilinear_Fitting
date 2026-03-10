"""
SiLU (Swish) lookup table for FP8 E4M3 quantization.
SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x)).
LUT[i] = FP8( SiLU( FP32( FP8_pattern i ) ) ) for i in 0..255.
"""

import numpy as np
from fp8_e4m3 import (
    fp8_e4m3_to_fp32_scalar,
    fp32_to_fp8_e4m3_scalar,
    fp8_e4m3_to_fp32,
    fp32_to_fp8_e4m3,
)


def silu(x: np.ndarray) -> np.ndarray:
    """SiLU(x) = x * sigmoid(x)."""
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -500, 500))))


def build_silu_fp8_lut() -> np.ndarray:
    """Build LUT of shape (256,) dtype uint8: LUT[fp8_input] = fp8_output."""
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        x = fp8_e4m3_to_fp32_scalar(i)
        if np.isnan(x):
            lut[i] = 0x00
            continue
        y = float(x) * (1.0 / (1.0 + np.exp(-np.clip(float(x), -500, 500))))
        lut[i] = fp32_to_fp8_e4m3_scalar(y) & 0xFF
    return lut


# Precomputed LUT (built at import)
SILU_FP8_LUT = build_silu_fp8_lut()


def silu_fp8_lut(fp8_input: np.ndarray) -> np.ndarray:
    """
    Apply SiLU via lookup table. Input and output are FP8 E4M3 (uint8).
    fp8_input: uint8 array (FP8 values as 0-255).
    Returns: uint8 array (FP8 SiLU output).
    """
    return np.take(SILU_FP8_LUT, fp8_input.ravel()).reshape(fp8_input.shape)


def silu_fp8_lut_to_fp32(fp8_input: np.ndarray) -> np.ndarray:
    """
    SiLU via LUT and return result as float32 (for error metrics).
    Input: FP8 uint8. Output: float32 (decoded from FP8 LUT output).
    """
    fp8_out = silu_fp8_lut(fp8_input)
    return fp8_e4m3_to_fp32(fp8_out)


if __name__ == "__main__":
    x = np.array([0.0, 1.0, 2.0, -1.0], dtype=np.float32)
    from fp8_e4m3 import fp32_to_fp8_e4m3
    x_fp8 = fp32_to_fp8_e4m3(x)
    y_fp8 = silu_fp8_lut(x_fp8)
    y_fp32 = fp8_e4m3_to_fp32(y_fp8)
    y_ref = silu(x)
    print("x:", x)
    print("silu_ref:", y_ref)
    print("silu_fp8_lut (decoded):", y_fp32)
    print("abs err:", np.abs(y_fp32 - y_ref))
