from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: patch_block_sparse_setup.py /path/to/setup.py")
    path = Path(sys.argv[1])
    text = path.read_text()
    if "arch=compute_86,code=sm_86" in text and "arch=compute_89,code=sm_89" in text:
        print(f"Block-Sparse gencodes already patched in {path}")
        return

    lines = text.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if "cc_flag +=" not in line or "arch=compute_80,code=sm_80" not in line:
            continue
        indent = line[: len(line) - len(line.lstrip())]
        insert = [
            f"{indent}# SceneBuilder approved 24-48 GB fleet: Ampere sm86 and Ada sm89.\n",
            f'{indent}if bare_metal_version >= Version("11.1") and "86" in archs:\n',
            f'{indent}    cc_flag += ["-gencode", "arch=compute_86,code=sm_86"]\n',
            f'{indent}if bare_metal_version >= Version("11.8") and "89" in archs:\n',
            f'{indent}    cc_flag += ["-gencode", "arch=compute_89,code=sm_89"]\n',
        ]
        lines[index + 1:index + 1] = insert
        path.write_text("".join(lines))
        print(f"patched Block-Sparse gencodes in {path}")
        return

    raise SystemExit(f"Block-Sparse sm80 gencode anchor missing: {path}")


if __name__ == "__main__":
    main()
