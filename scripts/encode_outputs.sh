#!/usr/bin/env bash
set -Eeuo pipefail

input="${1:?input video required}"
master_out="${2:?master output required}"
preview_out="${3:?preview output required}"
master_width="${4:?master width required}"
master_height="${5:?master height required}"
preview_width="${6:?preview width required}"
preview_height="${7:?preview height required}"
master_crf="${MASTER_CRF:-15}"
preview_crf="${PREVIEW_CRF:-20}"

mkdir -p "$(dirname "$master_out")" "$(dirname "$preview_out")"

ffmpeg -y -i "$input" \
  -map 0:v:0 -map 0:a? \
  -vf "crop=${master_width}:${master_height}:(iw-${master_width})/2:(ih-${master_height})/2" \
  -c:v libx265 -preset slow -crf "$master_crf" \
  -pix_fmt yuv420p \
  -c:a aac -b:a 192k \
  -tag:v hvc1 \
  -movflags +faststart \
  "$master_out"

ffmpeg -y -i "$input" \
  -map 0:v:0 -map 0:a? \
  -vf "crop=${master_width}:${master_height}:(iw-${master_width})/2:(ih-${master_height})/2,scale=${preview_width}:${preview_height}" \
  -c:v libx264 -preset veryfast -crf "$preview_crf" \
  -pix_fmt yuv420p \
  -c:a aac -b:a 128k \
  -movflags +faststart \
  "$preview_out"
