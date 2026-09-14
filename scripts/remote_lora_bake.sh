#!/usr/bin/env bash
set -Eeuo pipefail

: "${BAKE_COMPATIBILITY:?BAKE_COMPATIBILITY is required}"
: "${H3_BASE_IMAGE_TAG:?H3_BASE_IMAGE_TAG is required}"
: "${KREA2_BASE_IMAGE_TAG:?KREA2_BASE_IMAGE_TAG is required}"
: "${OUTPUT_IMAGE_TAG:?OUTPUT_IMAGE_TAG is required}"
: "${LORA_BAKE_REMOTE_DIR:?LORA_BAKE_REMOTE_DIR is required}"
: "${DOCKERHUB_USERNAME:?DOCKERHUB_USERNAME is required}"
: "${DOCKERHUB_TOKEN:?DOCKERHUB_TOKEN is required}"
: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"
: "${GITHUB_REF_NAME:?GITHUB_REF_NAME is required}"

export REGISTRY_NAMESPACE="${REGISTRY_NAMESPACE:-${DOCKERHUB_USERNAME}}"
repo_dir="/opt/scenebuilder-gpu-runtime"
docker_build_attempts="${DOCKER_BUILD_ATTEMPTS:-2}"
min_free_disk_gb="${MIN_FREE_DISK_GB:-45}"
release_manifest="${LORA_BAKE_REMOTE_DIR}/release-manifest.json"

log_disk() {
  if [ -x "${repo_dir}/scripts/log_disk.sh" ]; then
    "${repo_dir}/scripts/log_disk.sh" "$1"
  else
    echo "===== DISK USAGE: $1 ====="
    df -h || true
  fi
}

prune_build_cache_if_low() {
  local label="$1"
  local avail_kb threshold_kb
  avail_kb="$(df -Pk / | awk 'NR==2 {print $4}')"
  threshold_kb="$((min_free_disk_gb * 1024 * 1024))"
  if [ "${avail_kb:-0}" -ge "$threshold_kb" ]; then
    return 0
  fi
  echo "Free root disk below ${min_free_disk_gb} GiB before ${label}; pruning BuildKit cache."
  docker buildx prune --all --force || true
  sync || true
  log_disk "after prune ${label}"
}

install_docker() {
  if command -v docker >/dev/null 2>&1; then
    return
  fi
  apt-get update
  apt-get install -y --no-install-recommends ca-certificates curl git openssh-client
  curl -fsSL https://get.docker.com | sh
}

clone_repo() {
  rm -rf "${repo_dir}"
  if [ -n "${GH_FH_TOKEN_MM_H3_SERVERLESS:-}" ]; then
    git clone --depth 1 --branch "${GITHUB_REF_NAME}" \
      "https://x-access-token:${GH_FH_TOKEN_MM_H3_SERVERLESS}@github.com/${GITHUB_REPOSITORY}.git" \
      "${repo_dir}"
  else
    git clone --depth 1 --branch "${GITHUB_REF_NAME}" \
      "https://github.com/${GITHUB_REPOSITORY}.git" \
      "${repo_dir}"
  fi
}

image_for_target() {
  case "$1" in
    custom-nodes) echo "${REGISTRY_NAMESPACE}/minimax-h3-custom-nodes:${OUTPUT_IMAGE_TAG}" ;;
    fl2va-workflow) echo "${REGISTRY_NAMESPACE}/minimax-h3-fl2va-workflow:${OUTPUT_IMAGE_TAG}" ;;
    fl2va-loras) echo "${REGISTRY_NAMESPACE}/minimax-h3-fl2va-loras:${OUTPUT_IMAGE_TAG}" ;;
    runpod-fl2va) echo "${REGISTRY_NAMESPACE}/minimax-h3-runpod-fl2va:${OUTPUT_IMAGE_TAG}" ;;
    novita-fl2va) echo "${REGISTRY_NAMESPACE}/minimax-h3-novita-fl2va:${OUTPUT_IMAGE_TAG}" ;;
    ref2va-workflow) echo "${REGISTRY_NAMESPACE}/minimax-h3-ref2va-workflow:${OUTPUT_IMAGE_TAG}" ;;
    ref2va-loras) echo "${REGISTRY_NAMESPACE}/minimax-h3-ref2va-loras:${OUTPUT_IMAGE_TAG}" ;;
    runpod-ref2va) echo "${REGISTRY_NAMESPACE}/minimax-h3-runpod-ref2va:${OUTPUT_IMAGE_TAG}" ;;
    novita-ref2va) echo "${REGISTRY_NAMESPACE}/minimax-h3-novita-ref2va:${OUTPUT_IMAGE_TAG}" ;;
    pod-nodes) echo "${REGISTRY_NAMESPACE}/minimax-h3-pod-nodes:${OUTPUT_IMAGE_TAG}" ;;
    pod-loras) echo "${REGISTRY_NAMESPACE}/minimax-h3-pod-loras:${OUTPUT_IMAGE_TAG}" ;;
    pod-workflow) echo "${REGISTRY_NAMESPACE}/minimax-h3-pod-workflow:${OUTPUT_IMAGE_TAG}" ;;
    pod) echo "${REGISTRY_NAMESPACE}/minimax-h3-pod:${OUTPUT_IMAGE_TAG}" ;;
    krea2-system-adapters) echo "${REGISTRY_NAMESPACE}/scenebuilder-image-krea2-system-adapters:${OUTPUT_IMAGE_TAG}" ;;
    krea2-comfyui) echo "${REGISTRY_NAMESPACE}/scenebuilder-image-krea2-comfyui:${OUTPUT_IMAGE_TAG}" ;;
    krea2-nodes) echo "${REGISTRY_NAMESPACE}/scenebuilder-image-krea2-nodes:${OUTPUT_IMAGE_TAG}" ;;
    krea2-workflow) echo "${REGISTRY_NAMESPACE}/scenebuilder-image-krea2-workflow:${OUTPUT_IMAGE_TAG}" ;;
    krea2-user-loras) echo "${REGISTRY_NAMESPACE}/scenebuilder-image-krea2-user-loras:${OUTPUT_IMAGE_TAG}" ;;
    krea2-runtime) echo "${REGISTRY_NAMESPACE}/scenebuilder-image-krea2-runtime:${OUTPUT_IMAGE_TAG}" ;;
    *) echo "Unknown bake target: $1" >&2; return 1 ;;
  esac
}

dockerfile_for_target() {
  case "$1" in
    fl2va-loras) echo "${repo_dir}/docker/Dockerfile.fl2va-loras-bake" ;;
    ref2va-loras) echo "${repo_dir}/docker/Dockerfile.ref2va-loras-bake" ;;
    pod-loras) echo "${repo_dir}/docker/Dockerfile.pod-loras-bake" ;;
    krea2-user-loras) echo "${repo_dir}/docker/Dockerfile.krea2-user-loras" ;;
    krea2-runtime) echo "${repo_dir}/docker/Dockerfile.krea2-runtime-bake" ;;
    krea2-*) echo "${repo_dir}/docker/Dockerfile.${1}" ;;
    *) echo "${repo_dir}/docker/Dockerfile.${1}" ;;
  esac
}

append_target() {
  local target="$1"
  if [ -z "${seen_targets[$target]+x}" ]; then
    seen_targets["$target"]=1
    targets+=("$target")
  fi
}

append_h3_fl2va() {
  append_target custom-nodes
  append_target fl2va-workflow
  append_target fl2va-loras
  append_target runpod-fl2va
  append_target novita-fl2va
}

append_h3_ref2va() {
  append_target custom-nodes
  append_target ref2va-workflow
  append_target ref2va-loras
  append_target runpod-ref2va
  append_target novita-ref2va
}

append_h3_pod() {
  append_target custom-nodes
  append_target pod-nodes
  append_target pod-loras
  append_target pod-workflow
  append_target pod
}

append_krea2() {
  append_target krea2-system-adapters
  append_target krea2-comfyui
  append_target krea2-nodes
  append_target krea2-workflow
  append_target krea2-user-loras
  append_target krea2-runtime
}

expand_compatibility() {
  local raw compat
  declare -gA seen_targets=()
  declare -ga targets=()
  IFS=',' read -ra requested <<< "${BAKE_COMPATIBILITY}"
  for raw in "${requested[@]}"; do
    compat="$(echo "$raw" | xargs | tr '[:upper:]' '[:lower:]')"
    case "$compat" in
      h3-fl2va|fl2va) append_h3_fl2va ;;
      h3-ref2va|ref2va) append_h3_ref2va ;;
      h3-pod|pod) append_h3_pod ;;
      h3-all|h3) append_h3_fl2va; append_h3_ref2va; append_h3_pod ;;
      krea2|krea-2) append_krea2 ;;
      all) append_h3_fl2va; append_h3_ref2va; append_h3_pod; append_krea2 ;;
      "") ;;
      *) echo "Unknown compatibility value: ${compat}" >&2; exit 1 ;;
    esac
  done
  if [ "${#targets[@]}" -eq 0 ]; then
    echo "No bake targets selected" >&2
    exit 1
  fi
}

build_one_target() {
  local target="$1"
  local dockerfile image attempt status
  dockerfile="$(dockerfile_for_target "${target}")"
  image="$(image_for_target "${target}")"

  if [ ! -f "${dockerfile}" ]; then
    echo "Missing dockerfile: ${dockerfile}" >&2
    exit 1
  fi

  prune_build_cache_if_low "build ${target}"
  log_disk "before ${target}"

  attempt=1
  while true; do
    echo "===== LORA BAKE BUILD ${target}: attempt ${attempt}/${docker_build_attempts} ====="
    if docker buildx build \
      --file "${dockerfile}" \
      --tag "${image}" \
      --push \
      --progress plain \
      --build-arg "REGISTRY_NAMESPACE=${REGISTRY_NAMESPACE}" \
      --build-arg "IMAGE_TAG=${OUTPUT_IMAGE_TAG}" \
      --build-arg "BASE_IMAGE_TAG=${H3_BASE_IMAGE_TAG}" \
      --build-arg "COMFYUI_IMAGE_TAG=${H3_BASE_IMAGE_TAG}" \
      --build-arg "SAGEATTENTION_IMAGE_TAG=${H3_BASE_IMAGE_TAG}" \
      --build-arg "CUSTOM_NODES_IMAGE_TAG=${OUTPUT_IMAGE_TAG}" \
      --build-arg "FL2VA_BASE_IMAGE_TAG=${H3_BASE_IMAGE_TAG}" \
      --build-arg "REF2VA_BASE_IMAGE_TAG=${H3_BASE_IMAGE_TAG}" \
      --build-arg "POD_MODELS_IMAGE_TAG=${H3_BASE_IMAGE_TAG}" \
      --build-arg "KREA2_QWEN_IMAGE_TAG=${KREA2_BASE_IMAGE_TAG}" \
      --secret "id=hf_token,env=HF_TOKEN" \
      "${repo_dir}"; then
      status=0
    else
      status=$?
    fi

    if [ "${status}" -eq 0 ]; then
      break
    fi
    if [ "${attempt}" -ge "${docker_build_attempts}" ]; then
      echo "LoRA bake build failed after ${attempt} attempt(s): ${target}" >&2
      exit "${status}"
    fi
    docker buildx prune --all --force || true
    attempt=$((attempt + 1))
    sleep 15
  done

  echo "Pushed ${image}"
  pushed_images+=("${target}=${image}")
  log_disk "after ${target}"
}

write_release_manifest() {
  python3 - "$release_manifest" "$repo_dir/.lora-bake/manifest.json" "$H3_BASE_IMAGE_TAG" "$KREA2_BASE_IMAGE_TAG" "$OUTPUT_IMAGE_TAG" "${pushed_images[@]}" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
bake_manifest = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
h3_base = sys.argv[3]
krea2_base = sys.argv[4]
output_tag = sys.argv[5]
images = []
for raw in sys.argv[6:]:
    target, image = raw.split("=", 1)
    images.append({"target": target, "image": image})

payload = {
    "lora": bake_manifest,
    "h3BaseImageTag": h3_base,
    "krea2BaseImageTag": krea2_base,
    "outputImageTag": output_tag,
    "images": images,
}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(out)
PY
}

install_docker
clone_repo

mkdir -p "${repo_dir}/.lora-bake"
cp "${LORA_BAKE_REMOTE_DIR}/manifest.json" "${repo_dir}/.lora-bake/manifest.json"
cp "${LORA_BAKE_REMOTE_DIR}/lora.safetensors" "${repo_dir}/.lora-bake/lora.safetensors"
python3 "${repo_dir}/scripts/install_lora_bake.py" \
  --manifest "${repo_dir}/.lora-bake/manifest.json" \
  --file "${repo_dir}/.lora-bake/lora.safetensors" \
  --target-dir /tmp/lora-bake-verify \
  --manifest-out /tmp/lora-bake-verify/manifest.json
rm -rf /tmp/lora-bake-verify

echo "${DOCKERHUB_TOKEN}" | docker login -u "${DOCKERHUB_USERNAME}" --password-stdin
docker buildx create --name lora-bake-builder --use 2>/dev/null || docker buildx use lora-bake-builder

expand_compatibility
printf 'Selected bake targets:\n'
printf '  %s\n' "${targets[@]}"

declare -a pushed_images=()
for target in "${targets[@]}"; do
  build_one_target "${target}"
done

write_release_manifest
echo "LoRA bake release manifest written to ${release_manifest}"
