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


def get_silu_fp8_lut_curves():
    """
    Get (x_values, silu_ref, silu_quantized_decoded) for plotting.
    x_values are decoded FP8 inputs, sorted; silu_quantized_decoded is LUT output decoded to float.
    """
    from fp8_e4m3 import fp8_e4m3_to_fp32
    x_all = np.array([fp8_e4m3_to_fp32_scalar(i) for i in range(256)], dtype=np.float64)
    silu_ref = silu(x_all)
    silu_quant_fp8 = np.take(SILU_FP8_LUT, np.arange(256))
    silu_quant_decoded = fp8_e4m3_to_fp32(silu_quant_fp8)
    # Drop NaN inputs for a clean plot
    valid = ~(np.isnan(x_all) | np.isnan(silu_ref) | np.isnan(silu_quant_decoded))
    x_all = x_all[valid]
    silu_ref = silu_ref[valid]
    silu_quant_decoded = silu_quant_decoded[valid]
    sort_idx = np.argsort(x_all)
    x_values = x_all[sort_idx]
    silu_values = silu_ref[sort_idx]
    silu_quantized = silu_quant_decoded[sort_idx]
    return x_values, silu_values, silu_quantized


def plot_silu_function(x_values, silu_values, silu_quantized, scale_factor=1.0):
    """
    Plot SiLU function and quantization error (same style as impl_2).
    For FP8 LUT, pass scale_factor=1.0; silu_quantized should be decoded float.
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 5))

    # Left: SiLU float vs quantized
    plt.subplot(1, 2, 1)
    plt.plot(x_values, silu_values, "b-", linewidth=2, label="SiLU (float)")
    plt.plot(
        x_values,
        silu_quantized / scale_factor,
        "r--",
        linewidth=1,
        label="SiLU (FP8 LUT)",
    )
    plt.grid(True, alpha=0.3)
    plt.xlabel("Input x")
    plt.ylabel("SiLU(x) = x * sigmoid(x)")
    plt.title("SiLU activation (FP8 E4M3 LUT)")
    plt.legend()

    # Right: quantization error
    plt.subplot(1, 2, 2)
    error = silu_values - silu_quantized / scale_factor
    plt.plot(x_values, error * 100, "g-", linewidth=1)
    plt.grid(True, alpha=0.3)
    plt.xlabel("Input x")
    plt.ylabel("Error (%)")
    plt.title("Quantization error")

    plt.tight_layout()
    plt.savefig("./impl_1/silu_function_plot.png", dpi=150)
    plt.show()
    print("\nPlot saved: ./impl_1/silu_function_plot.png")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "plot":
        x_vals, silu_float, silu_quant = get_silu_fp8_lut_curves()
        print("Plotting SiLU FP8 LUT (impl_1)...")
        plot_silu_function(x_vals, silu_float, silu_quant, scale_factor=1.0)
        print("Max abs error:", np.max(np.abs(silu_float - silu_quant)))
        print("Mean abs error:", np.mean(np.abs(silu_float - silu_quant)))
    else:
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
        print("\nRun with 'plot' to generate silu_function_plot.png: python silu_lut.py plot")
