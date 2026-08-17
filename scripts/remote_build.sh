#!/usr/bin/env bash
set -Eeuo pipefail

: "${BUILD_TARGET:?BUILD_TARGET is required}"
: "${IMAGE_TAG:?IMAGE_TAG is required}"
: "${DOCKERHUB_USERNAME:?DOCKERHUB_USERNAME is required}"
: "${DOCKERHUB_TOKEN:?DOCKERHUB_TOKEN is required}"
: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"
: "${GITHUB_REF_NAME:?GITHUB_REF_NAME is required}"

export REGISTRY_NAMESPACE="${REGISTRY_NAMESPACE:-${DOCKERHUB_USERNAME}}"
repo_dir="/opt/minimax-h3-serverless"
docker_build_attempts="${DOCKER_BUILD_ATTEMPTS:-2}"
min_free_disk_gb="${MIN_FREE_DISK_GB:-45}"

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
  local avail_kb
  local threshold_kb

  avail_kb="$(df -Pk / | awk 'NR==2 {print $4}')"
  threshold_kb="$((min_free_disk_gb * 1024 * 1024))"
  if [ "${avail_kb:-0}" -ge "$threshold_kb" ]; then
    return 0
  fi

  echo "===== LOW DISK: ${label} ====="
  echo "Free root disk is below ${min_free_disk_gb} GiB; pruning pushed BuildKit cache."
  echo "All successful targets are already stored in Docker Hub, so this does not delete published images."
  docker buildx prune --all --force || true
  sync || true
  log_disk "after BuildKit prune ${label}"
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
  rm -rf "$repo_dir"
  if [ -n "${GH_FH_TOKEN_MM_H3_SERVERLESS:-}" ]; then
    git clone --depth 1 --branch "${GITHUB_REF_NAME}" "https://x-access-token:${GH_FH_TOKEN_MM_H3_SERVERLESS}@github.com/${GITHUB_REPOSITORY}.git" "$repo_dir"
  else
    git clone --depth 1 --branch "${GITHUB_REF_NAME}" "https://github.com/${GITHUB_REPOSITORY}.git" "$repo_dir"
  fi
}

image_for_target() {
  case "$1" in
    smoke) echo "${REGISTRY_NAMESPACE}/minimax-h3-smoke:${IMAGE_TAG}" ;;
    base) echo "${REGISTRY_NAMESPACE}/minimax-h3-base:${IMAGE_TAG}" ;;
    qwen-all) echo "${REGISTRY_NAMESPACE}/minimax-h3-qwen-all:${IMAGE_TAG}" ;;
    qwen-nvfp4) echo "${REGISTRY_NAMESPACE}/minimax-h3-qwen-nvfp4:${IMAGE_TAG}" ;;
    qwen-int8) echo "${REGISTRY_NAMESPACE}/minimax-h3-qwen-int8:${IMAGE_TAG}" ;;
    fl2va-base) echo "${REGISTRY_NAMESPACE}/minimax-h3-fl2va-base:${IMAGE_TAG}" ;;
    ref2va-base) echo "${REGISTRY_NAMESPACE}/minimax-h3-ref2va-base:${IMAGE_TAG}" ;;
    pod-models) echo "${REGISTRY_NAMESPACE}/minimax-h3-pod-models:${IMAGE_TAG}" ;;
    pod-nodes) echo "${REGISTRY_NAMESPACE}/minimax-h3-pod-nodes:${IMAGE_TAG}" ;;
    pod-loras) echo "${REGISTRY_NAMESPACE}/minimax-h3-pod-loras:${IMAGE_TAG}" ;;
    comfyui) echo "${REGISTRY_NAMESPACE}/minimax-h3-comfyui:${IMAGE_TAG}" ;;
    sageattention) echo "${REGISTRY_NAMESPACE}/minimax-h3-sageattention:${IMAGE_TAG}" ;;
    custom-nodes) echo "${REGISTRY_NAMESPACE}/minimax-h3-custom-nodes:${IMAGE_TAG}" ;;
    fl2va-workflow) echo "${REGISTRY_NAMESPACE}/minimax-h3-fl2va-workflow:${IMAGE_TAG}" ;;
    ref2va-workflow) echo "${REGISTRY_NAMESPACE}/minimax-h3-ref2va-workflow:${IMAGE_TAG}" ;;
    pod-workflow) echo "${REGISTRY_NAMESPACE}/minimax-h3-pod-workflow:${IMAGE_TAG}" ;;
    fl2va-loras) echo "${REGISTRY_NAMESPACE}/minimax-h3-fl2va-loras:${IMAGE_TAG}" ;;
    ref2va-loras) echo "${REGISTRY_NAMESPACE}/minimax-h3-ref2va-loras:${IMAGE_TAG}" ;;
    runpod-fl2va) echo "${REGISTRY_NAMESPACE}/minimax-h3-runpod-fl2va:${IMAGE_TAG}" ;;
    runpod-ref2va) echo "${REGISTRY_NAMESPACE}/minimax-h3-runpod-ref2va:${IMAGE_TAG}" ;;
    novita-fl2va) echo "${REGISTRY_NAMESPACE}/minimax-h3-novita-fl2va:${IMAGE_TAG}" ;;
    novita-ref2va) echo "${REGISTRY_NAMESPACE}/minimax-h3-novita-ref2va:${IMAGE_TAG}" ;;
    pod) echo "${REGISTRY_NAMESPACE}/minimax-h3-pod:${IMAGE_TAG}" ;;
    *) echo "Unknown BUILD_TARGET: $1" >&2; return 1 ;;
  esac
}

dockerfile_for_target() {
  echo "${repo_dir}/docker/Dockerfile.$1"
}

build_one_target() {
  local target="$1"
  local dockerfile
  local image
  local attempt
  local status

  dockerfile="$(dockerfile_for_target "${target}")"
  image="$(image_for_target "${target}")"

  if [ ! -f "$dockerfile" ]; then
    echo "Missing dockerfile: ${dockerfile}" >&2
    exit 1
  fi

  prune_build_cache_if_low "before docker build ${target}"
  log_disk "before docker build ${target}"

  attempt=1
  while true; do
    echo "Docker build attempt ${attempt}/${docker_build_attempts} for ${target}"
    if docker buildx build \
      --file "$dockerfile" \
      --tag "$image" \
      --push \
      --progress plain \
      --build-arg "REGISTRY_NAMESPACE=${REGISTRY_NAMESPACE}" \
      --build-arg "IMAGE_TAG=${IMAGE_TAG}" \
      --build-arg "BUILD_TARGET=${target}" \
      --secret "id=hf_token,env=HF_TOKEN" \
      "$repo_dir"; then
      status=0
    else
      status=$?
    fi

    if [ "$status" -eq 0 ]; then
      break
    fi

    log_disk "failed docker build ${target} attempt ${attempt}"
    prune_build_cache_if_low "failed build ${target} attempt ${attempt}"
    if [ "$attempt" -ge "$docker_build_attempts" ]; then
      echo "Docker build failed after ${attempt} attempt(s): ${target}" >&2
      exit "$status"
    fi
    attempt=$((attempt + 1))
    echo "Retrying ${target} in 15 seconds. Published parents will be pulled again if low-disk pruning removed local BuildKit cache."
    sleep 15
  done

  log_disk "after docker push ${target}"
  echo "Pushed ${image}"
  prune_build_cache_if_low "after docker push ${target}"
}

install_docker
log_disk "start before clone"
clone_repo
log_disk "after clone"

echo "${DOCKERHUB_TOKEN}" | docker login -u "${DOCKERHUB_USERNAME}" --password-stdin
docker buildx create --name h3builder --use 2>/dev/null || docker buildx use h3builder

targets_csv="${BUILD_TARGETS:-}"
if [ -z "${targets_csv// }" ]; then
  targets_csv="${BUILD_TARGET}"
fi

IFS=',' read -ra targets <<< "${targets_csv}"
for raw_target in "${targets[@]}"; do
  target="$(echo "${raw_target}" | xargs)"
  if [ -z "$target" ]; then
    continue
  fi
  echo "===== BUILD TARGET: ${target} ====="
  build_one_target "$target"
done
