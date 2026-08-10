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

log_disk() {
  if [ -x "${repo_dir}/scripts/log_disk.sh" ]; then
    "${repo_dir}/scripts/log_disk.sh" "$1"
  else
    echo "===== DISK USAGE: $1 ====="
    df -h || true
  fi
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
    fl2va-workflow) echo "${REGISTRY_NAMESPACE}/minimax-h3-fl2va-workflow:${IMAGE_TAG}" ;;
    ref2va-workflow) echo "${REGISTRY_NAMESPACE}/minimax-h3-ref2va-workflow:${IMAGE_TAG}" ;;
    fl2va-loras) echo "${REGISTRY_NAMESPACE}/minimax-h3-fl2va-loras:${IMAGE_TAG}" ;;
    ref2va-loras) echo "${REGISTRY_NAMESPACE}/minimax-h3-ref2va-loras:${IMAGE_TAG}" ;;
    runpod-fl2va) echo "${REGISTRY_NAMESPACE}/minimax-h3-runpod-fl2va:${IMAGE_TAG}" ;;
    runpod-ref2va) echo "${REGISTRY_NAMESPACE}/minimax-h3-runpod-ref2va:${IMAGE_TAG}" ;;
    novita-fl2va) echo "${REGISTRY_NAMESPACE}/minimax-h3-novita-fl2va:${IMAGE_TAG}" ;;
    novita-ref2va) echo "${REGISTRY_NAMESPACE}/minimax-h3-novita-ref2va:${IMAGE_TAG}" ;;
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

  dockerfile="$(dockerfile_for_target "${target}")"
  image="$(image_for_target "${target}")"

  if [ ! -f "$dockerfile" ]; then
    echo "Missing dockerfile: ${dockerfile}" >&2
    exit 1
  fi

  log_disk "before docker build ${target}"

  docker buildx build \
    --file "$dockerfile" \
    --tag "$image" \
    --push \
    --progress plain \
    --build-arg "REGISTRY_NAMESPACE=${REGISTRY_NAMESPACE}" \
    --build-arg "IMAGE_TAG=${IMAGE_TAG}" \
    --build-arg "BUILD_TARGET=${target}" \
    --secret "id=hf_token,env=HF_TOKEN" \
    "$repo_dir"

  log_disk "after docker push ${target}"
  echo "Pushed ${image}"
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
