"""
FP8 E4M3 format utilities (1 sign, 4 exponent, 3 mantissa; bias=7).
OFP8 / NVIDIA-style E4M3: no infinities, NaN = 0x7F/0xFF.
"""

import numpy as np


def fp32_to_fp8_e4m3_scalar(x: float) -> int:
    """Convert a single float to FP8 E4M3 (returns uint8 pattern 0-255)."""
    if np.isnan(x):
        return 0x7F
    if np.isinf(x):
        return 0x7E if x > 0 else 0xFE  # saturate to max magnitude
    if x == 0.0:
        return 0x00
    bits = np.float32(x).view(np.uint32).item()
    sign = (bits >> 31) & 1
    exp32 = ((bits >> 23) & 0xFF) - 127
    mant32 = bits & 0x7FFFFF
    if exp32 != -127:
        mant32 |= 0x800000
    e4m3_exp = exp32 + 7
    if e4m3_exp > 15:
        return (sign << 7) | 0x7E
    if e4m3_exp <= -3:
        return sign << 7
    if -3 < e4m3_exp <= 0:
        shift = 3 + e4m3_exp
        e4m3_mant = (mant32 >> (24 - shift)) & (0x7 >> (-e4m3_exp))
        return (sign << 7) | e4m3_mant
    e4m3_mant = (mant32 >> 20) & 0x7
    return (sign << 7) | (e4m3_exp << 3) | e4m3_mant


def fp8_e4m3_to_fp32_scalar(b: int) -> float:
    """Convert a single FP8 E4M3 byte (0-255) to float."""
    b = b & 0xFF
    if (b & 0x7F) == 0x7F:
        return np.nan
    sign = (b >> 7) & 1
    e4m3_exp = (b >> 3) & 0xF
    e4m3_mant = b & 0x7
    if e4m3_exp == 0 and e4m3_mant == 0:
        return -0.0 if sign else 0.0
    if e4m3_exp > 0:
        # normal: v = (-1)^s * 2^(e-7) * (1 + m/8)
        exp_val = int(e4m3_exp) - 7
        v = (1.0 + e4m3_mant / 8.0) * (2.0 ** exp_val)
    else:
        # subnormal: v = (-1)^s * 2^(1-7) * (m/8)
        v = (e4m3_mant / 8.0) * (2.0 ** -6)
    return -v if sign else v


_vec_fp32_to_fp8 = np.vectorize(fp32_to_fp8_e4m3_scalar)
_vec_fp8_to_fp32 = np.vectorize(fp8_e4m3_to_fp32_scalar)


def fp32_to_fp8_e4m3(x: np.ndarray) -> np.ndarray:
    """Convert float32 array to FP8 E4M3 (uint8 array)."""
    return _vec_fp32_to_fp8(x).astype(np.uint8)


def fp8_e4m3_to_fp32(b: np.ndarray) -> np.ndarray:
    """Convert FP8 E4M3 uint8 array to float32."""
    return _vec_fp8_to_fp32(b).astype(np.float32)


if __name__ == "__main__":
    # quick sanity
    for v in [0.0, 1.0, -1.0, 0.5, 448.0, 2**-6, 2**-9]:
        enc = fp32_to_fp8_e4m3_scalar(v)
        dec = fp8_e4m3_to_fp32_scalar(enc)
        print(f"  {v} -> 0x{enc:02x} -> {dec}")
