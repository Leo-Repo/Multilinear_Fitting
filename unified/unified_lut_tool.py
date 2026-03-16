#!/usr/bin/env python3
"""
Unified LUT/PWL generator for non-linear activations on fixed-point hardware.

Features:
- Unified frontend: function, quantization format, segment count.
- Three outputs:
  1) Full LUT
  2) PWL parameters (a, b, shift)
  3) Hardware files (.hex/.coe/.bin)
- Unified evaluation:
  - Approximation errors
  - Resource estimates (ROM, multiplier width, rough latency)
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np


def silu(x: np.ndarray) -> np.ndarray:
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0))))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def tanh_fn(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


ACTIVATIONS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "silu": silu,
    "sigmoid": sigmoid,
    "tanh": tanh_fn,
    "relu": relu,
}


def fp32_to_fp8_e4m3_scalar(x: float) -> int:
    if np.isnan(x):
        return 0x7F
    if np.isinf(x):
        return 0x7E if x > 0 else 0xFE
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
    b = b & 0xFF
    if (b & 0x7F) == 0x7F:
        return float("nan")
    sign = (b >> 7) & 1
    e4m3_exp = (b >> 3) & 0xF
    e4m3_mant = b & 0x7
    if e4m3_exp == 0 and e4m3_mant == 0:
        return -0.0 if sign else 0.0
    if e4m3_exp > 0:
        exp_val = int(e4m3_exp) - 7
        v = (1.0 + e4m3_mant / 8.0) * (2.0 ** exp_val)
    else:
        v = (e4m3_mant / 8.0) * (2.0 ** -6)
    return -v if sign else v


def fp32_to_fp8_e4m3(arr: np.ndarray) -> np.ndarray:
    vec = np.vectorize(fp32_to_fp8_e4m3_scalar)
    return vec(arr).astype(np.uint8)


def fp8_e4m3_to_fp32(arr: np.ndarray) -> np.ndarray:
    vec = np.vectorize(fp8_e4m3_to_fp32_scalar)
    return vec(arr).astype(np.float32)


@dataclass
class QuantFormat:
    kind: str
    total_bits: int = 32
    frac_bits: int = 9
    signed: bool = True
    name: str = "q1_22_9"


def parse_quant_format(text: str) -> QuantFormat:
    text_l = text.lower()
    if text_l == "fp8_e4m3":
        return QuantFormat(kind="fp8", total_bits=8, frac_bits=0, signed=True, name="fp8_e4m3")
    # Common integer aliases.
    if text_l == "int8":
        return QuantFormat(kind="fixed", total_bits=8, frac_bits=0, signed=True, name="int8")
    if text_l == "uint8":
        return QuantFormat(kind="fixed", total_bits=8, frac_bits=0, signed=False, name="uint8")
    if text_l == "int16":
        return QuantFormat(kind="fixed", total_bits=16, frac_bits=0, signed=True, name="int16")
    if text_l == "int32":
        return QuantFormat(kind="fixed", total_bits=32, frac_bits=0, signed=True, name="int32")

    if text_l.startswith("q"):
        # q<signed>_<int>_<frac>, ex: q1_22_9
        body = text_l[1:]
        parts = body.split("_")
        if len(parts) != 3:
            raise ValueError(f"Invalid q format: {text}")
        sign_bits = int(parts[0])
        int_bits = int(parts[1])
        frac_bits = int(parts[2])
        total_bits = sign_bits + int_bits + frac_bits
        if sign_bits not in (0, 1):
            raise ValueError(f"Invalid sign bits in {text}")
        return QuantFormat(
            kind="fixed",
            total_bits=total_bits,
            frac_bits=frac_bits,
            signed=(sign_bits == 1),
            name=text_l,
        )
    raise ValueError(f"Unsupported format: {text}")


def quantize_fixed(x: np.ndarray, fmt: QuantFormat) -> np.ndarray:
    scale = 2 ** fmt.frac_bits
    q = np.round(x * scale).astype(np.int64)
    if fmt.signed:
        qmin = -(2 ** (fmt.total_bits - 1))
        qmax = 2 ** (fmt.total_bits - 1) - 1
    else:
        qmin = 0
        qmax = 2 ** fmt.total_bits - 1
    return np.clip(q, qmin, qmax).astype(np.int64)


def dequantize_fixed(q: np.ndarray, fmt: QuantFormat) -> np.ndarray:
    return q.astype(np.float64) / (2 ** fmt.frac_bits)


def quantize_values(x: np.ndarray, fmt: QuantFormat) -> np.ndarray:
    if fmt.kind == "fp8":
        return fp32_to_fp8_e4m3(x.astype(np.float32)).astype(np.int64)
    return quantize_fixed(x, fmt)


def dequantize_values(q: np.ndarray, fmt: QuantFormat) -> np.ndarray:
    if fmt.kind == "fp8":
        return fp8_e4m3_to_fp32(q.astype(np.uint8)).astype(np.float64)
    return dequantize_fixed(q.astype(np.int64), fmt)


def full_lut_domain(input_fmt: QuantFormat, lut_size: int, input_range: float) -> Tuple[np.ndarray, np.ndarray]:
    if input_fmt.kind == "fp8":
        idx = np.arange(256, dtype=np.int64)
        x = dequantize_values(idx, input_fmt)
        nan_mask = np.isnan(x)
        x[nan_mask] = 0.0
        return idx, x
    idx = np.arange(lut_size, dtype=np.int64)
    x = np.linspace(-input_range, input_range, lut_size, dtype=np.float64)
    return idx, x


def to_unsigned_word(v: int, bits: int) -> int:
    mask = (1 << bits) - 1
    return v & mask


def save_words(words: np.ndarray, out_path: str, bits: int, fmt: str) -> None:
    width_hex = max(1, math.ceil(bits / 4))
    with open(out_path, "w", encoding="utf-8") as f:
        if fmt == "hex":
            for i, w in enumerate(words):
                f.write(f"{to_unsigned_word(int(w), bits):0{width_hex}X}  // idx {i}\n")
        elif fmt == "bin":
            for i, w in enumerate(words):
                f.write(f"{to_unsigned_word(int(w), bits):0{bits}b}  // idx {i}\n")
        elif fmt == "coe":
            f.write("memory_initialization_radix=16;\n")
            f.write("memory_initialization_vector=\n")
            for i, w in enumerate(words):
                end = ";" if i == len(words) - 1 else ","
                f.write(f"{to_unsigned_word(int(w), bits):0{width_hex}X}{end}\n")
        else:
            raise ValueError(f"Unsupported hw format: {fmt}")


@dataclass
class PwlParams:
    breakpoints: np.ndarray
    slope_q: np.ndarray
    bias_q: np.ndarray
    shift: np.ndarray
    slope_scale_bits: int
    bias_frac_bits: int


def fit_pwl(
    fn: Callable[[np.ndarray], np.ndarray],
    input_range: float,
    segments: int,
    slope_bits: int,
    bias_bits: int,
    bias_frac_bits: int,
    shift_bits: int,
    eval_points_per_segment: int = 256,
) -> PwlParams:
    bps = np.linspace(-input_range, input_range, segments + 1, dtype=np.float64)
    slope_q = np.zeros(segments, dtype=np.int64)
    bias_q = np.zeros(segments, dtype=np.int64)
    shift = np.zeros(segments, dtype=np.int64)
    slope_max = (2 ** (slope_bits - 1)) - 1
    slope_min = -(2 ** (slope_bits - 1))
    bias_max = (2 ** (bias_bits - 1)) - 1
    bias_min = -(2 ** (bias_bits - 1))
    max_shift = (2 ** shift_bits) - 1

    for i in range(segments):
        x0, x1 = bps[i], bps[i + 1]
        xs = np.linspace(x0, x1, eval_points_per_segment, dtype=np.float64)
        ys = fn(xs)
        A = np.vstack([xs, np.ones_like(xs)]).T
        a_float, b_float = np.linalg.lstsq(A, ys, rcond=None)[0]

        best = None
        for s in range(max_shift + 1):
            a_scaled = int(np.round(a_float * (2 ** s)))
            a_scaled = int(np.clip(a_scaled, slope_min, slope_max))
            b_scaled = int(np.round(b_float * (2 ** bias_frac_bits)))
            b_scaled = int(np.clip(b_scaled, bias_min, bias_max))

            y_hat = (a_scaled / (2 ** s)) * xs + (b_scaled / (2 ** bias_frac_bits))
            err = np.mean((y_hat - ys) ** 2)
            if best is None or err < best[0]:
                best = (err, a_scaled, b_scaled, s)

        assert best is not None
        slope_q[i] = best[1]
        bias_q[i] = best[2]
        shift[i] = best[3]

    return PwlParams(
        breakpoints=bps,
        slope_q=slope_q,
        bias_q=bias_q,
        shift=shift,
        slope_scale_bits=slope_bits,
        bias_frac_bits=bias_frac_bits,
    )


def pwl_apply(x: np.ndarray, params: PwlParams) -> np.ndarray:
    seg_idx = np.clip(np.searchsorted(params.breakpoints, x, side="right") - 1, 0, len(params.slope_q) - 1)
    a = params.slope_q[seg_idx].astype(np.float64)
    b = params.bias_q[seg_idx].astype(np.float64)
    s = params.shift[seg_idx].astype(np.float64)
    return (a / np.power(2.0, s)) * x + b / (2 ** params.bias_frac_bits)


def evaluate(
    fn: Callable[[np.ndarray], np.ndarray],
    input_range: float,
    full_lut_x: np.ndarray,
    full_lut_y_q: np.ndarray,
    output_fmt: QuantFormat,
    pwl_params: PwlParams,
    eval_samples: int = 20000,
) -> Dict[str, float]:
    xs = np.linspace(-input_range, input_range, eval_samples, dtype=np.float64)
    y_ref = fn(xs)

    # PWL
    y_pwl = pwl_apply(xs, pwl_params)
    mae_pwl = float(np.mean(np.abs(y_pwl - y_ref)))
    maxe_pwl = float(np.max(np.abs(y_pwl - y_ref)))
    mse_pwl = float(np.mean((y_pwl - y_ref) ** 2))

    # Full LUT (nearest sample index for fixed domain)
    idx = np.searchsorted(full_lut_x, xs, side="left")
    idx = np.clip(idx, 0, len(full_lut_x) - 1)
    y_lut = dequantize_values(full_lut_y_q[idx], output_fmt)
    mae_lut = float(np.mean(np.abs(y_lut - y_ref)))
    maxe_lut = float(np.max(np.abs(y_lut - y_ref)))
    mse_lut = float(np.mean((y_lut - y_ref) ** 2))

    return {
        "pwl_mae": mae_pwl,
        "pwl_max_abs_err": maxe_pwl,
        "pwl_mse": mse_pwl,
        "lut_mae": mae_lut,
        "lut_max_abs_err": maxe_lut,
        "lut_mse": mse_lut,
    }


def estimate_resources(
    lut_entries: int,
    output_bits: int,
    segments: int,
    slope_bits: int,
    bias_bits: int,
    shift_bits: int,
    input_compute_bits: int,
) -> Dict[str, int]:
    lut_rom_bits = lut_entries * output_bits
    pwl_rom_bits = segments * (slope_bits + bias_bits + shift_bits)
    return {
        "full_lut_rom_bits": int(lut_rom_bits),
        "pwl_param_rom_bits": int(pwl_rom_bits),
        "multiplier_input_bits": int(input_compute_bits),
        "multiplier_slope_bits": int(slope_bits),
        "multiplier_total_bits": int(input_compute_bits + slope_bits),
        "latency_cycles_lut_est": 1,
        "latency_cycles_pwl_est": 3,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified LUT/PWL generation tool.")
    parser.add_argument("--function", default="silu", choices=sorted(ACTIVATIONS.keys()))
    parser.add_argument("--input-format", default="fp8_e4m3")
    parser.add_argument("--output-format", default="fp8_e4m3")
    parser.add_argument(
        "--compute-format",
        default="q1_22_9",
        help="Internal compute format (e.g., q1_22_9, int32, int8, fp8_e4m3).",
    )
    parser.add_argument("--segments", type=int, default=16)
    parser.add_argument("--input-range", type=float, default=8.0)
    parser.add_argument("--lut-size", type=int, default=256)
    parser.add_argument("--slope-bits", type=int, default=8)
    parser.add_argument("--bias-bits", type=int, default=16)
    parser.add_argument("--bias-frac-bits", type=int, default=9)
    parser.add_argument("--shift-bits", type=int, default=4)
    parser.add_argument(
        "--hw-formats",
        nargs="+",
        default=["hex", "coe", "bin"],
        choices=["hex", "coe", "bin"],
        help="Hardware file formats to generate. Default: hex coe bin",
    )
    parser.add_argument("--out-dir", default="unified/out")
    args = parser.parse_args()

    fn = ACTIVATIONS[args.function]
    input_fmt = parse_quant_format(args.input_format)
    output_fmt = parse_quant_format(args.output_format)
    compute_fmt = parse_quant_format(args.compute_format)

    os.makedirs(args.out_dir, exist_ok=True)

    # Full LUT
    lut_idx, lut_x = full_lut_domain(input_fmt, args.lut_size, args.input_range)
    lut_y = fn(lut_x)
    lut_y_q = quantize_values(lut_y, output_fmt)

    # PWL
    pwl = fit_pwl(
        fn=fn,
        input_range=args.input_range,
        segments=args.segments,
        slope_bits=args.slope_bits,
        bias_bits=args.bias_bits,
        bias_frac_bits=args.bias_frac_bits,
        shift_bits=args.shift_bits,
    )

    # Eval
    eval_result = evaluate(
        fn=fn,
        input_range=args.input_range,
        full_lut_x=np.sort(lut_x),
        full_lut_y_q=lut_y_q[np.argsort(lut_x)],
        output_fmt=output_fmt,
        pwl_params=pwl,
    )
    output_bits = output_fmt.total_bits
    input_compute_bits = compute_fmt.total_bits if compute_fmt.kind == "fixed" else 32
    resource = estimate_resources(
        lut_entries=len(lut_y_q),
        output_bits=output_bits,
        segments=args.segments,
        slope_bits=args.slope_bits,
        bias_bits=args.bias_bits,
        shift_bits=args.shift_bits,
        input_compute_bits=input_compute_bits,
    )

    # Save artifacts
    base = f"{args.function}_{input_fmt.name}_to_{output_fmt.name}_via_{compute_fmt.name}"
    for hw_fmt in args.hw_formats:
        save_words(
            lut_y_q,
            os.path.join(args.out_dir, f"{base}_full_lut.{hw_fmt}"),
            output_bits,
            hw_fmt,
        )
    np.savetxt(os.path.join(args.out_dir, f"{base}_full_lut_values.txt"), lut_y_q.astype(np.int64), fmt="%d")

    pwl_obj = {
        "breakpoints": pwl.breakpoints.tolist(),
        "slope_q": pwl.slope_q.tolist(),
        "bias_q": pwl.bias_q.tolist(),
        "shift": pwl.shift.tolist(),
        "slope_bits": args.slope_bits,
        "bias_bits": args.bias_bits,
        "bias_frac_bits": args.bias_frac_bits,
        "shift_bits": args.shift_bits,
    }
    with open(os.path.join(args.out_dir, f"{base}_pwl_params.json"), "w", encoding="utf-8") as f:
        json.dump(pwl_obj, f, indent=2)

    # Save compact packed PWL words for hw ROM
    packed = []
    for a, b, s in zip(pwl.slope_q, pwl.bias_q, pwl.shift):
        aw = to_unsigned_word(int(a), args.slope_bits)
        bw = to_unsigned_word(int(b), args.bias_bits)
        sw = to_unsigned_word(int(s), args.shift_bits)
        packed_word = (aw << (args.bias_bits + args.shift_bits)) | (bw << args.shift_bits) | sw
        packed.append(packed_word)
    packed_bits = args.slope_bits + args.bias_bits + args.shift_bits
    packed_arr = np.array(packed, dtype=np.int64)
    for hw_fmt in args.hw_formats:
        save_words(
            packed_arr,
            os.path.join(args.out_dir, f"{base}_pwl_params.{hw_fmt}"),
            packed_bits,
            hw_fmt,
        )

    report = {
        "config": vars(args),
        "eval": eval_result,
        "resource_estimate": resource,
    }
    report_path = os.path.join(args.out_dir, f"{base}_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Done.")
    print(f"Output dir: {args.out_dir}")
    print(f"Report: {report_path}")
    print("Eval:")
    for k, v in eval_result.items():
        print(f"  {k}: {v:.8f}")
    print("Resource estimate:")
    for k, v in resource.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
