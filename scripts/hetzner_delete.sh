#!/usr/bin/env bash
set -Eeuo pipefail

: "${HETZNER_TOKEN:?HETZNER_TOKEN is required}"
server_id="${1:?server id required}"
ssh_key_id="${2:-}"

echo "Deleting Hetzner server ${server_id}"
curl -fsS \
  -X DELETE \
  -H "Authorization: Bearer ${HETZNER_TOKEN}" \
  "https://api.hetzner.cloud/v1/servers/${server_id}" >/dev/null
echo "Delete request accepted for ${server_id}"

if [ -n "${ssh_key_id}" ]; then
  echo "Deleting Hetzner SSH key ${ssh_key_id}"
  curl -fsS \
    -X DELETE \
    -H "Authorization: Bearer ${HETZNER_TOKEN}" \
    "https://api.hetzner.cloud/v1/ssh_keys/${ssh_key_id}" >/dev/null || true
  echo "SSH key delete request accepted for ${ssh_key_id}"
fi
