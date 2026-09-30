"""Convert a torch.save'd state dict to .npz for model-version-diff.

PyTorch's .pth/.pt format is a Python pickle, which model-version-diff
deliberately does not parse without torch installed. If you have torch,
either point the manifest straight at the .pth file (torch is imported
lazily) or use this script to convert to .npz first::

    python examples/convert_torch_to_npz.py model_v1.pth model_v1.npz

Only plain state dicts (or full checkpoints with a "state_dict" key) are
supported; anything else raises a clear error.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: convert_torch_to_npz.py <input.pth> <output.npz>")
        return 2
    try:
        import torch
    except ImportError:
        print("error: this script needs torch installed (pip install torch)", file=sys.stderr)
        return 1

    src, dst = Path(argv[1]), Path(argv[2])
    obj = torch.load(src, map_location="cpu", weights_only=True)
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        print(f"error: {src} does not look like a state dict", file=sys.stderr)
        return 1
    arrays = {}
    for name, value in obj.items():
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        arrays[str(name)] = np.asarray(value)
    np.savez(dst, **arrays)
    print(f"wrote {len(arrays)} tensors to {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
