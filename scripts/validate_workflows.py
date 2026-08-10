#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_path(obj, dotted: str):
    current = obj
    for part in dotted.split("."):
        if isinstance(current, list):
            current = current[int(part)]
        elif isinstance(current, dict):
            if part not in current:
                raise KeyError(part)
            current = current[part]
        else:
            raise KeyError(part)
    return current


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate workflow JSON and manifest node paths.")
    parser.add_argument("--workflow", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    workflow_path = Path(args.workflow)
    manifest_path = Path(args.manifest)
    workflow = load_json(workflow_path)
    manifest = load_json(manifest_path)

    missing = []
    for label, path in manifest.get("requiredPaths", {}).items():
        try:
            resolve_path(workflow, path)
        except Exception:
            missing.append(f"{label}: {path}")

    if missing:
        print("Workflow manifest points to missing paths:", file=sys.stderr)
        for item in missing:
            print(f"  - {item}", file=sys.stderr)
        return 1

    print(f"OK workflow={workflow_path.name} manifest={manifest_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
