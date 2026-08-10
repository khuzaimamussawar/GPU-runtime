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

user_data_path="$work_dir/cloud-init.yaml"
payload_path="$work_dir/create-server.json"
response_path="$work_dir/create-server-response.json"

cat > "$user_data_path" <<EOF
#cloud-config
users:
  - name: root
    ssh_authorized_keys:
      - ${BUILDER_SSH_PUBLIC_KEY}
ssh_pwauth: false
disable_root: false
package_update: false
package_upgrade: false
EOF

python3 - "$payload_path" "$user_data_path" <<'PY'
import json
import os
import sys
payload_path, user_data_path = sys.argv[1], sys.argv[2]
payload = {
    "name": os.environ["BUILDER_NAME"],
    "server_type": os.environ["SERVER_TYPE"],
    "image": os.environ["HETZNER_IMAGE"],
    "location": os.environ["HETZNER_LOCATION"],
    "start_after_create": True,
    "user_data": open(user_data_path, "r", encoding="utf-8").read(),
    "public_net": {
        "enable_ipv4": True,
        "enable_ipv6": False
    }
}
with open(payload_path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle)
PY

echo "Creating Hetzner builder ${BUILDER_NAME} (${SERVER_TYPE}, ${HETZNER_LOCATION})"
curl -fsS \
  -H "Authorization: Bearer ${HETZNER_TOKEN}" \
  -H "Content-Type: application/json" \
  -d "@${payload_path}" \
  https://api.hetzner.cloud/v1/servers > "$response_path"

server_id="$(python3 -c "import json; print(json.load(open('$response_path'))['server']['id'])")"
server_ip="$(python3 -c "import json; print(json.load(open('$response_path'))['server']['public_net']['ipv4']['ip'])")"

echo "server_id=${server_id}"
echo "server_ip=${server_ip}"

if [ -n "${GITHUB_OUTPUT:-}" ]; then
  {
    echo "server_id=${server_id}"
    echo "server_ip=${server_ip}"
  } >> "$GITHUB_OUTPUT"
fi
