# Unified Activation LUT/PWL Tool

This folder provides one unified flow to generate activation approximation artifacts for hardware:

1. Full LUT
2. PWL parameters (`a, b, shift`)
3. Hardware initialization files (`.hex`, `.coe`, `.bin`)
4. Unified evaluation report (error + resource estimate)

## Supported frontend options

- Activation function: `silu`, `sigmoid`, `tanh`, `relu`
- Input format:
  - `fp8_e4m3`
  - `int8`, `uint8`, `int16`, `int32`
  - fixed-point Q format string, e.g. `q1_22_9`
- Output format:
  - `fp8_e4m3`
  - `int8`, `uint8`, `int16`, `int32`
  - fixed-point Q format string
- Compute format:
  - `fp8_e4m3`
  - `int8`, `uint8`, `int16`, `int32`
  - fixed-point Q format string
- Segment count: any positive integer (commonly 16)

## Quick start

Run from project root:

```bash
python unified/unified_lut_tool.py \
  --function silu \
  --input-format fp8_e4m3 \
  --compute-format q1_22_9 \
  --output-format fp8_e4m3 \
  --segments 16 \
  --input-range 8.0 \
  --lut-size 256 \
  --slope-bits 8 \
  --bias-bits 16 \
  --bias-frac-bits 9 \
  --shift-bits 4 \
  --hw-formats hex coe bin \
  --out-dir unified/out
```

## Example for INT32 Q1.22.9 compute semantics

If your compute path uses INT32 with Q1.22.9 interpretation:

```bash
python unified/unified_lut_tool.py \
  --function silu \
  --input-format q1_22_9 \
  --compute-format q1_22_9 \
  --output-format q1_22_9 \
  --segments 16 \
  --input-range 8.0 \
  --lut-size 256 \
  --slope-bits 8 \
  --bias-bits 16 \
  --bias-frac-bits 9 \
  --shift-bits 4 \
  --hw-formats hex coe bin \
  --out-dir unified/out_q1229
```

## Example for INT8 input + INT32 compute

```bash
python unified/unified_lut_tool.py \
  --function silu \
  --input-format int8 \
  --compute-format q1_22_9 \
  --output-format int8 \
  --segments 16 \
  --input-range 8.0 \
  --lut-size 256 \
  --hw-formats hex coe bin \
  --out-dir unified/out_int8
```

## Output files

For base name `<function>_<input>_to_<output>`:

- `<base>_full_lut.<hex|coe|bin>`
- `<base>_full_lut_values.txt`
- `<base>_pwl_params.json`
- `<base>_pwl_params.<hex|coe|bin>` (packed ROM words)
- `<base>_report.json`

## Notes

- `--slope-bits 8` is often used to reduce multiplier cost and ROM bandwidth.
- `--segments 16` is common because 4-bit segment index is hardware-friendly.
- Error in report compares approximation against float reference on a dense grid.
