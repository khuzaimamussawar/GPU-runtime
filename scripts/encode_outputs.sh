#!/usr/bin/env bash
set -Eeuo pipefail

input="${1:?input video required}"
master_out="${2:?master output required}"
preview_out="${3:?preview output required}"
preview_scale="${4:-854:-2}"
master_crf="${MASTER_CRF:-15}"
preview_crf="${PREVIEW_CRF:-20}"

mkdir -p "$(dirname "$master_out")" "$(dirname "$preview_out")"

ffmpeg -y -i "$input" \
  -c:v libx265 -preset slow -crf "$master_crf" \
  -pix_fmt yuv420p \
  -tag:v hvc1 \
  -movflags +faststart \
  "$master_out"

ffmpeg -y -i "$input" \
  -vf "scale=${preview_scale}" \
  -c:v libx264 -preset veryfast -crf "$preview_crf" \
  -pix_fmt yuv420p \
  -movflags +faststart \
  "$preview_out"
