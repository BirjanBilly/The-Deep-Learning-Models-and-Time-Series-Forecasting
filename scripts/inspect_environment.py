from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect CPU/GPU capabilities before training")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    payload = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_compiled": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        "bfloat16_supported": bool(
            torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        ),
        "torch_compile_available": hasattr(torch, "compile"),
        "tf32_matmul_enabled": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_available": torch.backends.cudnn.is_available(),
    }
    text = json.dumps(payload, indent=2)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
