from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi, snapshot_download

RIFE_REPO = "Steffen097/rife-4.9"
RIFE_EXPECTED_FLOWNET_SHA256 = "ef91580a020abb7ddfbd3a51573dc395cf2c2a9530ff653ef3f8a1fc6845857f"
GIMM_REPO = "GSean/GIMM-VFI"
FLASHVSR_REPO = "JunhaoZhuang/FlashVSR-v1.1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_snapshot(repo_id: str, destination: Path, *, allow_patterns: Iterable[str] | None = None) -> dict:
    token = os.environ.get("HF_TOKEN") or None
    api = HfApi(token=token)
    info = api.model_info(repo_id)
    revision = str(info.sha)
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(destination),
        token=token,
        allow_patterns=list(allow_patterns) if allow_patterns else None,
    )
    metadata = {"repo": repo_id, "revision": revision}
    (destination / ".scenebuilder-source.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def download_rife(root: Path) -> dict:
    destination = root / "rife-4.9"
    meta = download_snapshot(RIFE_REPO, destination)
    flownet = destination / "flownet.pkl"
    if not flownet.is_file():
        raise RuntimeError(f"{RIFE_REPO} did not contain flownet.pkl")
    actual = sha256(flownet)
    if actual != RIFE_EXPECTED_FLOWNET_SHA256:
        raise RuntimeError(f"RIFE 4.9 checkpoint checksum mismatch: {actual}")
    meta["flownet_sha256"] = actual
    (destination / ".scenebuilder-source.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def download_gimm(root: Path) -> dict:
    destination = root / "gimm-vfi"
    return download_snapshot(GIMM_REPO, destination)


def download_flashvsr(root: Path) -> dict:
    destination = root / "flashvsr-v1.1"
    return download_snapshot(FLASHVSR_REPO, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("group", choices=["vfi", "flashvsr", "all"])
    parser.add_argument("--root", default="/opt/scenebuilder-models")
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    result: dict[str, dict] = {}
    if args.group in {"vfi", "all"}:
        result["rife-4.9"] = download_rife(root)
        result["gimm-vfi-f"] = download_gimm(root)
    if args.group in {"flashvsr", "all"}:
        result["flashvsr-v1.1"] = download_flashvsr(root)

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
