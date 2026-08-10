#!/usr/bin/env bash
set -Eeuo pipefail

input="${1:?input video required}"
output="${2:?output video required}"
mode="${3:-artifactreduction}"

mkdir -p "$(dirname "$output")"

provider="${SCENEBUILDER_VFX_CLEANUP_PROVIDER:-maxine}"
enabled="${SCENEBUILDER_VFX_CLEANUP_ENABLED:-0}"
cleanup_bin="${SCENEBUILDER_VFX_CLEANUP_BIN:-}"
supported_gpus="${SCENEBUILDER_VFX_SUPPORTED_GPUS:-NVIDIA L40,NVIDIA L4,NVIDIA A10,NVIDIA A16,NVIDIA A30,NVIDIA A40,NVIDIA A100,NVIDIA H100,NVIDIA A2,NVIDIA T4,NVIDIA B40,NVIDIA B100,NVIDIA B200}"

copy_input() {
  cp -f "$input" "$output"
}

if [[ "$enabled" != "1" ]]; then
  echo "Artifact cleanup disabled in runtime; passing source through unchanged."
  copy_input
  exit 0
fi

if [[ "$provider" != "maxine" ]]; then
  echo "Unsupported cleanup provider '$provider'; passing source through unchanged."
  copy_input
  exit 0
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not available; passing source through unchanged."
  copy_input
  exit 0
fi

gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 | tr -d '\r')"
if [[ -z "$gpu_name" ]]; then
  echo "GPU detection returned empty name; passing source through unchanged."
  copy_input
  exit 0
fi

is_supported=0
IFS=',' read -ra gpu_list <<< "$supported_gpus"
for supported in "${gpu_list[@]}"; do
  supported="$(echo "$supported" | xargs)"
  if [[ -n "$supported" && "$gpu_name" == *"$supported"* ]]; then
    is_supported=1
    break
  fi
done

if [[ "$is_supported" != "1" ]]; then
  echo "GPU '$gpu_name' is not in the conservative VFX support allowlist; passing source through unchanged."
  copy_input
  exit 0
fi

if [[ -z "$cleanup_bin" || ! -x "$cleanup_bin" ]]; then
  echo "No Maxine/VFX cleanup binary configured; passing source through unchanged."
  copy_input
  exit 0
fi

echo "Running optional Maxine/VFX cleanup on GPU '$gpu_name' with mode '$mode'."
"$cleanup_bin" --input "$input" --output "$output" --mode "$mode"

