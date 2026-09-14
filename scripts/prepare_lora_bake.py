#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import hmac
import json
import os
import re
import shutil
import sys
import urllib.parse
import urllib.request
from pathlib import Path


SAFE_FILE_RE = re.compile(r"^[A-Za-z0-9._-]+\.safetensors$")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_object_key(value: str) -> str:
    raw = value.strip()
    if not raw:
        raise SystemExit("r2_object_key_or_url is required")
    if raw.startswith("http://") or raw.startswith("https://"):
        parsed = urllib.parse.urlparse(raw)
        path = urllib.parse.unquote(parsed.path.lstrip("/"))
        bucket = os.environ.get("R2_BUCKET_NAME", "").strip()
        if bucket and path.startswith(bucket + "/"):
            path = path[len(bucket) + 1 :]
        raw = path
    raw = raw.lstrip("/")
    if not raw.startswith("models/lora/"):
        raise SystemExit("LoRA R2 object key must begin with models/lora/")
    if ".." in raw.split("/"):
        raise SystemExit("LoRA R2 object key must not contain .. segments")
    return raw


def s3_signing_key(secret: str, date_stamp: str, region: str, service: str) -> bytes:
    key_date = hmac.new(("AWS4" + secret).encode(), date_stamp.encode(), hashlib.sha256).digest()
    key_region = hmac.new(key_date, region.encode(), hashlib.sha256).digest()
    key_service = hmac.new(key_region, service.encode(), hashlib.sha256).digest()
    return hmac.new(key_service, b"aws4_request", hashlib.sha256).digest()


def download_r2_object(object_key: str, target: Path) -> None:
    endpoint = (os.environ.get("R2_ENDPOINT_URL") or os.environ.get("R2_ENDPOINT") or "").strip().rstrip("/")
    bucket = os.environ.get("R2_BUCKET_NAME", "").strip()
    access_key = (os.environ.get("R2_ACCESS_KEY_ID") or os.environ.get("R2_ACCESS_KEY") or "").strip()
    secret_key = (os.environ.get("R2_SECRET_ACCESS_KEY") or os.environ.get("R2_SECRET_KEY") or "").strip()
    region = os.environ.get("R2_REGION", "auto").strip() or "auto"

    missing = [
        name
        for name, value in {
            "R2_ENDPOINT_URL or R2_ENDPOINT": endpoint,
            "R2_BUCKET_NAME": bucket,
            "R2_ACCESS_KEY_ID or R2_ACCESS_KEY": access_key,
            "R2_SECRET_ACCESS_KEY or R2_SECRET_KEY": secret_key,
        }.items()
        if not value
    ]
    if missing:
        raise SystemExit(f"Missing R2 secret/env values: {', '.join(missing)}")

    parsed = urllib.parse.urlparse(endpoint)
    if parsed.scheme not in {"https", "http"} or not parsed.netloc:
        raise SystemExit("R2_ENDPOINT_URL must be an http(s) endpoint")

    encoded_key = "/".join(urllib.parse.quote(part, safe="") for part in object_key.split("/"))
    canonical_uri = f"/{bucket}/{encoded_key}"
    url = endpoint + canonical_uri

    now = dt.datetime.now(dt.timezone.utc)
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = now.strftime("%Y%m%d")
    payload_hash = hashlib.sha256(b"").hexdigest()
    host = parsed.netloc
    canonical_headers = (
        f"host:{host}\n"
        f"x-amz-content-sha256:{payload_hash}\n"
        f"x-amz-date:{amz_date}\n"
    )
    signed_headers = "host;x-amz-content-sha256;x-amz-date"
    canonical_request = "\n".join(
        ["GET", canonical_uri, "", canonical_headers, signed_headers, payload_hash]
    )
    credential_scope = f"{date_stamp}/{region}/s3/aws4_request"
    string_to_sign = "\n".join(
        [
            "AWS4-HMAC-SHA256",
            amz_date,
            credential_scope,
            hashlib.sha256(canonical_request.encode()).hexdigest(),
        ]
    )
    signature = hmac.new(
        s3_signing_key(secret_key, date_stamp, region, "s3"),
        string_to_sign.encode(),
        hashlib.sha256,
    ).hexdigest()
    authorization = (
        "AWS4-HMAC-SHA256 "
        f"Credential={access_key}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )

    request = urllib.request.Request(
        url,
        headers={
            "Host": host,
            "X-Amz-Date": amz_date,
            "X-Amz-Content-SHA256": payload_hash,
            "Authorization": authorization,
        },
        method="GET",
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(request, timeout=120) as response, target.open("wb") as output:
        shutil.copyfileobj(response, output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify a LoRA bake input.")
    parser.add_argument("--lora-id", required=True)
    parser.add_argument("--r2-object-key-or-url", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--expected-size-bytes", required=True, type=int)
    parser.add_argument("--file-name", required=True)
    parser.add_argument("--compatibility", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    lora_id = args.lora_id.strip()
    file_name = args.file_name.strip()
    expected_sha = args.expected_sha256.strip().lower()
    expected_size = args.expected_size_bytes
    object_key = normalize_object_key(args.r2_object_key_or_url)

    if not lora_id:
        raise SystemExit("lora_id is required")
    if not SAFE_FILE_RE.fullmatch(file_name):
        raise SystemExit("file_name must be a safe .safetensors basename")
    if not SHA_RE.fullmatch(expected_sha):
        raise SystemExit("expected_sha256 must be a lowercase 64-character SHA-256")
    if expected_size <= 0:
        raise SystemExit("expected_size_bytes must be positive")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    lora_path = args.out_dir / "lora.safetensors"
    manifest_path = args.out_dir / "manifest.json"

    download_r2_object(object_key, lora_path)
    actual_size = lora_path.stat().st_size
    if actual_size != expected_size:
        raise SystemExit(f"Downloaded LoRA size mismatch: {actual_size} != {expected_size}")
    actual_sha = sha256_file(lora_path)
    if actual_sha != expected_sha:
        raise SystemExit(f"Downloaded LoRA sha256 mismatch: {actual_sha} != {expected_sha}")

    manifest = {
        "loraId": lora_id,
        "r2ObjectKey": object_key,
        "fileName": file_name,
        "sha256": expected_sha,
        "fileSizeBytes": expected_size,
        "compatibility": [item.strip() for item in args.compatibility.split(",") if item.strip()],
    }
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "file": str(lora_path), "sha256": actual_sha}))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"prepare_lora_bake failed: {exc}", file=sys.stderr)
        raise
