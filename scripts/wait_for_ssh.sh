#!/usr/bin/env bash
set -Eeuo pipefail

host="${1:?host required}"
key_path="${2:?ssh key path required}"
timeout_seconds="${3:-420}"
deadline=$((SECONDS + timeout_seconds))

echo "Waiting for SSH on ${host}"
attempt=0
while [ "$SECONDS" -lt "$deadline" ]; do
  attempt=$((attempt + 1))
  if ssh -i "$key_path" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=8 "root@${host}" "echo ready" 2>/dev/null | grep -q ready; then
    echo "SSH ready on ${host}"
    exit 0
  fi
  if [ $((attempt % 5)) -eq 0 ]; then
    echo "Still waiting for SSH on ${host} after ${SECONDS}s..."
  fi
  sleep 8
done

echo "SSH did not become ready within ${timeout_seconds}s" >&2
exit 1
