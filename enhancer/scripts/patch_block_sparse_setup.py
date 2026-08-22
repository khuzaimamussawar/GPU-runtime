from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: patch_block_sparse_setup.py /path/to/setup.py")
    path = Path(sys.argv[1])
    text = path.read_text()
    old = """    # Always-regular 80
    if "80" in archs:
        cc_flag += ["-gencode", "arch=compute_80,code=sm_80"]
    # Hopper 9.0 needs >= 11.8
    if bare_metal_version >= Version("11.8") and "90" in archs:
        cc_flag += ["-gencode", "arch=compute_90,code=sm_90"]
"""
    new = """    # SceneBuilder approved 24-48 GB fleet: Ampere sm86, Ada sm89, Blackwell RTX sm120.
    if "80" in archs:
        cc_flag += ["-gencode", "arch=compute_80,code=sm_80"]
    if bare_metal_version >= Version("11.1") and "86" in archs:
        cc_flag += ["-gencode", "arch=compute_86,code=sm_86"]
    if bare_metal_version >= Version("11.8") and "89" in archs:
        cc_flag += ["-gencode", "arch=compute_89,code=sm_89"]
    # Hopper 9.0 needs >= 11.8
    if bare_metal_version >= Version("11.8") and "90" in archs:
        cc_flag += ["-gencode", "arch=compute_90,code=sm_90"]
"""
    if old not in text:
        raise SystemExit(f"Block-Sparse setup.py gencode anchor missing: {path}")
    path.write_text(text.replace(old, new))
    print(f"patched Block-Sparse gencodes in {path}")


if __name__ == "__main__":
    main()
