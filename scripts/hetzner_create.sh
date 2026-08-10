#!/usr/bin/env bash
set -Eeuo pipefail

: "${HETZNER_TOKEN:?HETZNER_TOKEN is required}"
: "${BUILDER_NAME:?BUILDER_NAME is required}"
: "${SERVER_TYPE:?SERVER_TYPE is required}"
: "${HETZNER_IMAGE:?HETZNER_IMAGE is required}"
: "${HETZNER_LOCATION:?HETZNER_LOCATION is required}"
: "${BUILDER_SSH_PUBLIC_KEY:?BUILDER_SSH_PUBLIC_KEY is required}"

work_dir="${RUNNER_TEMP:-/tmp}/h3-hetzner"
mkdir -p "$work_dir"

payload_path="$work_dir/create-server.json"
response_path="$work_dir/create-server-response.json"
ssh_key_payload_path="$work_dir/create-ssh-key.json"
ssh_key_response_path="$work_dir/create-ssh-key-response.json"

python3 - "$ssh_key_payload_path" <<'PY'
import json
import os
import sys
payload_path = sys.argv[1]
payload = {
    "name": f"{os.environ['BUILDER_NAME']}-key",
    "public_key": os.environ["BUILDER_SSH_PUBLIC_KEY"],
}
with open(payload_path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle)
PY

echo "Creating Hetzner SSH key ${BUILDER_NAME}-key"
ssh_key_status="$(curl -sS \
  -w "%{http_code}" \
  -o "$ssh_key_response_path" \
  -H "Authorization: Bearer ${HETZNER_TOKEN}" \
  -H "Content-Type: application/json" \
  -d "@${ssh_key_payload_path}" \
  https://api.hetzner.cloud/v1/ssh_keys)"

if [ "${ssh_key_status}" -lt 200 ] || [ "${ssh_key_status}" -ge 300 ]; then
  echo "Hetzner create SSH key failed with HTTP ${ssh_key_status}" >&2
  echo "Response body:" >&2
  python3 -m json.tool "$ssh_key_response_path" >&2 || cat "$ssh_key_response_path" >&2
  exit 1
fi

ssh_key_id="$(python3 -c "import json; print(json.load(open('$ssh_key_response_path'))['ssh_key']['id'])")"

python3 - "$payload_path" "$ssh_key_id" <<'PY'
import json
import os
import sys
payload_path, ssh_key_id = sys.argv[1], int(sys.argv[2])
payload = {
    "name": os.environ["BUILDER_NAME"],
    "server_type": os.environ["SERVER_TYPE"],
    "image": os.environ["HETZNER_IMAGE"],
    "location": os.environ["HETZNER_LOCATION"],
    "start_after_create": True,
    "ssh_keys": [ssh_key_id],
    "public_net": {
        "enable_ipv4": True,
        "enable_ipv6": False
    }
}
with open(payload_path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle)
PY

echo "Creating Hetzner builder ${BUILDER_NAME} (${SERVER_TYPE}, ${HETZNER_LOCATION})"
http_status="$(curl -sS \
  -w "%{http_code}" \
  -o "$response_path" \
  -H "Authorization: Bearer ${HETZNER_TOKEN}" \
  -H "Content-Type: application/json" \
  -d "@${payload_path}" \
  https://api.hetzner.cloud/v1/servers)"

if [ "${http_status}" -lt 200 ] || [ "${http_status}" -ge 300 ]; then
  echo "Hetzner create server failed with HTTP ${http_status}" >&2
  echo "Response body:" >&2
  python3 -m json.tool "$response_path" >&2 || cat "$response_path" >&2
  exit 1
fi

server_id="$(python3 -c "import json; print(json.load(open('$response_path'))['server']['id'])")"
server_ip="$(python3 -c "import json; print(json.load(open('$response_path'))['server']['public_net']['ipv4']['ip'])")"

echo "server_id=${server_id}"
echo "server_ip=${server_ip}"
echo "ssh_key_id=${ssh_key_id}"

if [ -n "${GITHUB_OUTPUT:-}" ]; then
  {
    echo "server_id=${server_id}"
    echo "server_ip=${server_ip}"
    echo "ssh_key_id=${ssh_key_id}"
  } >> "$GITHUB_OUTPUT"
fi
