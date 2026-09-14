#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path


SAFE_FILE_RE = re.compile(r"^[A-Za-z0-9._-]+\.safetensors$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Install a verified baked LoRA into a runtime image.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--file", required=True, type=Path)
    parser.add_argument("--target-dir", required=True, type=Path)
    parser.add_argument("--manifest-out", required=True, type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    file_name = str(manifest.get("fileName") or "").strip()
    expected_sha = str(manifest.get("sha256") or "").strip().lower()
    expected_size = int(manifest.get("fileSizeBytes") or 0)
    lora_id = str(manifest.get("loraId") or "").strip()

    if not lora_id:
        raise SystemExit("LoRA bake manifest is missing loraId")
    if not SAFE_FILE_RE.fullmatch(file_name):
        raise SystemExit(f"Unsafe LoRA file name: {file_name!r}")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
        raise SystemExit("LoRA bake manifest has invalid sha256")
    if expected_size <= 0:
        raise SystemExit("LoRA bake manifest has invalid fileSizeBytes")
    if not args.file.is_file():
        raise SystemExit(f"LoRA file does not exist: {args.file}")
    if args.file.stat().st_size != expected_size:
        raise SystemExit(
            f"LoRA size mismatch: {args.file.stat().st_size} != {expected_size}"
        )

    actual_sha = sha256_file(args.file)
    if actual_sha != expected_sha:
        raise SystemExit(f"LoRA sha256 mismatch: {actual_sha} != {expected_sha}")

    args.target_dir.mkdir(parents=True, exist_ok=True)
    target = args.target_dir / file_name
    shutil.copyfile(args.file, target)

    installed_sha = sha256_file(target)
    if installed_sha != expected_sha:
        raise SystemExit(f"Installed LoRA sha256 mismatch: {installed_sha} != {expected_sha}")

    record = dict(manifest)
    record["bakedPath"] = str(target)
    record["installedSha256"] = installed_sha
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(f"Installed baked LoRA {lora_id} at {target}")


if __name__ == "__main__":
    main()
