#!/usr/bin/env bash
set -Eeuo pipefail

host="${1:?host required}"
key_path="${2:?ssh key path required}"
timeout_seconds="${3:-420}"
deadline=$((SECONDS + timeout_seconds))

echo "Waiting for SSH on ${host}"
while [ "$SECONDS" -lt "$deadline" ]; do
  if ssh -i "$key_path" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=8 "root@${host}" "echo ready" 2>/dev/null | grep -q ready; then
    echo "SSH ready on ${host}"
    exit 0
  fi
  sleep 8
done

echo "SSH did not become ready within ${timeout_seconds}s" >&2
exit 1
