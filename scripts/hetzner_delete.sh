#!/usr/bin/env bash
set -Eeuo pipefail

: "${HETZNER_TOKEN:?HETZNER_TOKEN is required}"
server_id="${1:?server id required}"

echo "Deleting Hetzner server ${server_id}"
curl -fsS \
  -X DELETE \
  -H "Authorization: Bearer ${HETZNER_TOKEN}" \
  "https://api.hetzner.cloud/v1/servers/${server_id}" >/dev/null
echo "Delete request accepted for ${server_id}"
