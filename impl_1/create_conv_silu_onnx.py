"""
Create a minimal Conv2d + SiLU ONNX model for FP8 quantization testing.
Run once to generate conv_silu.onnx. Requires torch and onnx.
"""

import os

try:
    import torch
    import torch.nn as nn
except ImportError:
    raise SystemExit("Need torch to create ONNX. Install: pip install torch")

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "conv_silu.onnx")


class ConvSiLU(nn.Module):
    def __init__(self, in_c=2, out_c=4, k=3):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, padding=k // 2)

    def forward(self, x):
        return torch.nn.functional.silu(self.conv(x))


def main():
    model = ConvSiLU(in_c=2, out_c=4, k=3)
    model.eval()
    dummy = torch.randn(1, 2, 8, 8)
    torch.onnx.export(
        model,
        dummy,
        OUTPUT_PATH,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
