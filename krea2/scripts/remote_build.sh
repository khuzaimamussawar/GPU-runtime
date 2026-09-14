#!/usr/bin/env bash
set -Eeuo pipefail

: "${BUILD_TARGET:?BUILD_TARGET is required}"
: "${IMAGE_TAG:?IMAGE_TAG is required}"
: "${DOCKERHUB_USERNAME:?DOCKERHUB_USERNAME is required}"
: "${DOCKERHUB_TOKEN:?DOCKERHUB_TOKEN is required}"
: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"
: "${GITHUB_REF_NAME:?GITHUB_REF_NAME is required}"

export REGISTRY_NAMESPACE="${REGISTRY_NAMESPACE:-${DOCKERHUB_USERNAME}}"
repo_dir="/opt/scenebuilder-gpu-runtime"
docker_build_attempts="${DOCKER_BUILD_ATTEMPTS:-2}"
min_free_disk_gb="${MIN_FREE_DISK_GB:-45}"
provider_disk_gb="${IMAGE_POD_DISK_GB:-40}"
min_runtime_headroom_gb="${MIN_RUNTIME_HEADROOM_GB:-5}"
krea2_qwen_image_tag="${KREA2_QWEN_IMAGE_TAG:-${IMAGE_TAG}}"

ALL_TARGETS=(
  base
  model-int8
  vaes
  qwen-bf16
  system-adapters
  comfyui
  nodes
  workflow
  runtime
)

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
    base) echo "${REGISTRY_NAMESPACE}/scenebuilder-image-krea2-base:${IMAGE_TAG}" ;;
    model-int8) echo "${REGISTRY_NAMESPACE}/scenebuilder-image-krea2-model-int8:${IMAGE_TAG}" ;;
    vaes) echo "${REGISTRY_NAMESPACE}/scenebuilder-image-krea2-vaes:${IMAGE_TAG}" ;;
    qwen-bf16) echo "${REGISTRY_NAMESPACE}/scenebuilder-image-krea2-qwen-bf16:${IMAGE_TAG}" ;;
    system-adapters) echo "${REGISTRY_NAMESPACE}/scenebuilder-image-krea2-system-adapters:${IMAGE_TAG}" ;;
    comfyui) echo "${REGISTRY_NAMESPACE}/scenebuilder-image-krea2-comfyui:${IMAGE_TAG}" ;;
    nodes) echo "${REGISTRY_NAMESPACE}/scenebuilder-image-krea2-nodes:${IMAGE_TAG}" ;;
    workflow) echo "${REGISTRY_NAMESPACE}/scenebuilder-image-krea2-workflow:${IMAGE_TAG}" ;;
    runtime) echo "${REGISTRY_NAMESPACE}/scenebuilder-image-krea2-runtime:${IMAGE_TAG}" ;;
    *) echo "Unknown Krea2 target: $1" >&2; return 1 ;;
  esac
}

dockerfile_for_target() {
  echo "${repo_dir}/docker/Dockerfile.krea2-$1"
}

expand_targets() {
  local requested="$1"
  if [ "${requested}" = "all" ]; then
    printf '%s\n' "${ALL_TARGETS[@]}"
    return
  fi
  printf '%s\n' "${requested}"
}

verify_runtime_image_manifest() {
  local image="$1"
  local inspect_file="/tmp/krea2-runtime-imagetools.txt"

  if [ "${provider_disk_gb}" -le "${min_runtime_headroom_gb}" ]; then
    echo "Invalid image disk/headroom configuration: ${provider_disk_gb} GiB disk, ${min_runtime_headroom_gb} GiB headroom" >&2
    exit 1
  fi

  echo "===== VERIFY KREA2 RUNTIME REGISTRY MANIFEST ====="
  echo "Target provider container disk: ${provider_disk_gb} GiB"
  echo "Required runtime writable headroom: ${min_runtime_headroom_gb} GiB"
  echo "The CPX22 builder will NOT docker-pull/extract the final runtime image for a virtual-size check."
  echo "Pulling the just-pushed multi-layer image duplicates/extracts the full CUDA/model stack and can exhaust builder disk even when the push itself succeeded."
  echo "Exact writable-headroom validation must be performed during the real provider GPU canary after the container disk has been created."

  rm -f "${inspect_file}"
  docker buildx imagetools inspect "${image}" > "${inspect_file}"
  sed -n '1,80p' "${inspect_file}"
  echo "Krea2 runtime registry manifest verified without local layer extraction."
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
    echo "===== KREA2 BUILD ${target}: attempt ${attempt}/${docker_build_attempts} ====="
    if docker buildx build \
      --file "${dockerfile}" \
      --tag "${image}" \
      --push \
      --progress plain \
      --build-arg "REGISTRY_NAMESPACE=${REGISTRY_NAMESPACE}" \
      --build-arg "IMAGE_TAG=${IMAGE_TAG}" \
      --build-arg "KREA2_QWEN_IMAGE_TAG=${krea2_qwen_image_tag}" \
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
      echo "Krea2 build failed after ${attempt} attempt(s): ${target}" >&2
      exit "${status}"
    fi
    docker buildx prune --all --force || true
    attempt=$((attempt + 1))
    sleep 15
  done

  echo "Pushed ${image}"
  if [ "${target}" = "runtime" ]; then
    verify_runtime_image_manifest "${image}"
  fi
  log_disk "after ${target}"
  prune_build_cache_if_low "after ${target}"
}

install_docker
clone_repo

echo "${DOCKERHUB_TOKEN}" | docker login -u "${DOCKERHUB_USERNAME}" --password-stdin
docker buildx create --name krea2builder --use 2>/dev/null || docker buildx use krea2builder

requested="${BUILD_TARGETS:-${BUILD_TARGET}}"
IFS=',' read -ra raw_targets <<< "${requested}"
for raw in "${raw_targets[@]}"; do
  raw="$(echo "${raw}" | xargs)"
  [ -n "${raw}" ] || continue
  while IFS= read -r target; do
    [ -n "${target}" ] || continue
    build_one_target "${target}"
  done < <(expand_targets "${raw}")
done
