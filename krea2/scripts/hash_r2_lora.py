#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from typing import Iterable


CANONICAL_LORA_KEY_RE = re.compile(r"^models/lora/(?:krea2|h3|shared)/[A-Za-z0-9._-]+/[A-Za-z0-9._-]+\.safetensors$")


def required_env(name: str) -> str:
    value = str(os.environ.get(name, "")).strip()
    if not value:
        raise SystemExit(f"missing required environment variable: {name}")
    return value


def r2_client():
    try:
        import boto3
    except ImportError as exc:
        raise SystemExit("boto3 is required: pip install boto3") from exc

    return boto3.client(
        "s3",
        endpoint_url=required_env("R2_ENDPOINT"),
        aws_access_key_id=required_env("R2_ACCESS_KEY"),
        aws_secret_access_key=required_env("R2_SECRET_KEY"),
        region_name=os.environ.get("R2_REGION", "auto"),
    )


def hash_object(client, bucket: str, key: str, chunk_size: int = 8 * 1024 * 1024) -> dict[str, object]:
    response = client.get_object(Bucket=bucket, Key=key)
    declared_size = int(response.get("ContentLength") or 0)
    digest = hashlib.sha256()
    actual_size = 0
    body = response["Body"]
    try:
        while True:
            chunk = body.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
            actual_size += len(chunk)
    finally:
        body.close()

    if declared_size and actual_size != declared_size:
        raise RuntimeError(
            f"size mismatch while hashing {key}: R2={declared_size}, streamed={actual_size}"
        )

    return {
        "r2ObjectKey": key,
        "fileSizeBytes": actual_size,
        "sha256": digest.hexdigest(),
    }


def sql_escape(value: str) -> str:
    return value.replace("'", "''")


def emit_sql(result: dict[str, object]) -> str:
    key = sql_escape(str(result["r2ObjectKey"]))
    sha = sql_escape(str(result["sha256"]))
    size = int(result["fileSizeBytes"])
    timestamp = int(time.time() * 1000)
    return (
        "UPDATE lora "
        f"SET file_size_bytes = {size}, sha256 = '{sha}', updated_at = {timestamp} "
        f"WHERE r2_object_key = '{key}';"
    )


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream existing LoRA objects from R2 and compute exact SHA256 + byte size."
    )
    parser.add_argument(
        "--key",
        action="append",
        required=True,
        help="R2 object key. Repeat --key for multiple LoRAs.",
    )
    parser.add_argument(
        "--sql",
        action="store_true",
        help="Also print D1 UPDATE statements keyed by r2_object_key.",
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    bucket = required_env("R2_BUCKET_NAME")
    client = r2_client()

    for key in args.key:
        clean_key = str(key or "").strip().lstrip("/")
        if not CANONICAL_LORA_KEY_RE.fullmatch(clean_key):
            raise SystemExit(f"refusing non-canonical LoRA key: {clean_key}")
        result = hash_object(client, bucket, clean_key)
        print(json.dumps(result, separators=(",", ":")))
        if args.sql:
            print(emit_sql(result))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
