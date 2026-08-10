#!/usr/bin/env bash
set -Eeuo pipefail

label="${1:-snapshot}"

echo ""
echo "===== DISK USAGE: ${label} ====="
date -u +"%Y-%m-%dT%H:%M:%SZ"
df -h || true
docker system df || true
du -sh /var/lib/docker 2>/dev/null || true
du -sh /tmp 2>/dev/null || true
echo "===== END DISK USAGE: ${label} ====="
echo ""
