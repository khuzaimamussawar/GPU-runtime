#!/usr/bin/env python3
import argparse
import os
import sys
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser(description="Download one Hugging Face file into a Docker layer.")
    parser.add_argument("--token-env", default=None)
    parser.add_argument("--token-file", default=None)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--file", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.repo.startswith("__TODO"):
        raise SystemExit(
            f"HF repo is still TODO for {args.file}. Set the real repo before building this target."
        )

    token = None
    if args.token_file and os.path.exists(args.token_file):
        with open(args.token_file, "r", encoding="utf-8") as handle:
            token = handle.read().strip()
    if not token and args.token_env:
        token = os.environ.get(args.token_env)
    if not token:
        raise SystemExit("Missing required Hugging Face token")

    url = f"https://huggingface.co/{args.repo}/resolve/main/{args.file}"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    tmp = f"{args.out}.part"
    print(f"Downloading {args.file} -> {args.out}", flush=True)
    with urllib.request.urlopen(request, timeout=120) as response, open(tmp, "wb") as handle:
        while True:
            chunk = response.read(1024 * 1024 * 8)
            if not chunk:
                break
            handle.write(chunk)
    os.replace(tmp, args.out)
    print("Download complete", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
