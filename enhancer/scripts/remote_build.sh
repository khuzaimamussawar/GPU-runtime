#!/usr/bin/env bash
set -Eeuo pipefail

: "${BUILD_TARGET:?BUILD_TARGET is required}"
: "${IMAGE_TAG:?IMAGE_TAG is required}"
: "${DOCKERHUB_USERNAME:?DOCKERHUB_USERNAME is required}"
: "${DOCKERHUB_TOKEN:?DOCKERHUB_TOKEN is required}"
: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"
: "${GITHUB_REF_NAME:?GITHUB_REF_NAME is required}"

export REGISTRY_NAMESPACE="${REGISTRY_NAMESPACE:-${DOCKERHUB_USERNAME}}"
repo_dir="/opt/scenebuilder-gpu-runtime-enhancer"
context_dir="${repo_dir}/enhancer"
docker_build_attempts="${DOCKER_BUILD_ATTEMPTS:-2}"
min_free_disk_gb="${MIN_FREE_DISK_GB:-35}"
previous_workflow_run_id="${PREVIOUS_WORKFLOW_RUN_ID:-}"
COMMON_TARGETS=(smoke base torch vfi-models)
FAST_TARGETS=(esrgan-models fast)
QUALITY_TARGETS=(flashvsr-runtime flashvsr-models quality)
ALL_TARGETS=("${COMMON_TARGETS[@]}" "${FAST_TARGETS[@]}" "${QUALITY_TARGETS[@]}")

valid_target() {
  case "$1" in
    smoke|base|torch|vfi-models|esrgan-models|fast|flashvsr-runtime|flashvsr-models|quality) return 0 ;;
    *) return 1 ;;
  esac
}

image_for_target() {
  echo "${REGISTRY_NAMESPACE}/scenebuilder-enhancer-$1:${IMAGE_TAG}"
}

image_exists() {
  local target="$1" image
  image="$(image_for_target "$target")"
  docker manifest inspect "$image" >/dev/null 2>&1
}

append_target() {
  local target="$1"
  valid_target "$target" || { echo "Unknown enhancer target: $target" >&2; exit 2; }
  local existing
  for existing in "${EXPANDED_TARGETS[@]}"; do
    [ "$existing" = "$target" ] && return
  done
  EXPANDED_TARGETS+=("$target")
}

append_targets() {
  local target
  for target in "$@"; do
    append_target "$target"
  done
}

append_all_targets() { append_targets "${ALL_TARGETS[@]}"; }
append_fast_targets() { append_targets "${COMMON_TARGETS[@]}" "${FAST_TARGETS[@]}"; }
append_quality_targets() { append_targets "${COMMON_TARGETS[@]}" "${QUALITY_TARGETS[@]}"; }

append_remaining_targets() {
  local target image
  if [ -n "$previous_workflow_run_id" ]; then
    echo "Resume requested from previous workflow run id: ${previous_workflow_run_id}"
  else
    echo "Resume requested without previous_workflow_run_id; using Docker Hub image presence for ${IMAGE_TAG}."
  fi
  for target in "${ALL_TARGETS[@]}"; do
    image="$(image_for_target "$target")"
    if image_exists "$target"; then
      echo "Skipping already-published layer: ${image}"
    else
      echo "Remaining layer missing: ${image}"
      append_target "$target"
    fi
  done
}

dockerfile_for_target() {
  echo "${context_dir}/docker/Dockerfile.$1"
}

log_disk() {
  echo "===== ENHANCER DISK: $1 ====="
  df -h || true
  docker system df || true
}

prune_if_low() {
  local avail_kb threshold_kb
  avail_kb="$(df -Pk / | awk 'NR==2 {print $4}')"
  threshold_kb="$((min_free_disk_gb * 1024 * 1024))"
  if [ "${avail_kb:-0}" -lt "$threshold_kb" ]; then
    echo "Low disk; pruning local BuildKit cache. Published parent images remain in Docker Hub."
    docker buildx prune --all --force || true
    sync || true
  fi
}

install_docker() {
  command -v docker >/dev/null 2>&1 && return
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

build_one_target() {
  local target="$1" dockerfile image attempt status
  valid_target "$target" || { echo "Unknown enhancer target: $target" >&2; exit 2; }
  dockerfile="$(dockerfile_for_target "$target")"
  image="$(image_for_target "$target")"
  test -f "$dockerfile" || { echo "Missing dockerfile: $dockerfile" >&2; exit 1; }
  prune_if_low
  log_disk "before ${target}"

  attempt=1
  while true; do
    echo "===== ENHANCER BUILD ${target}: attempt ${attempt}/${docker_build_attempts} ====="
    if docker buildx build \
      --file "$dockerfile" \
      --tag "$image" \
      --push \
      --progress plain \
      --build-arg "REGISTRY_NAMESPACE=${REGISTRY_NAMESPACE}" \
      --build-arg "IMAGE_TAG=${IMAGE_TAG}" \
      --secret "id=hf_token,env=HF_TOKEN" \
      "$context_dir"; then
      status=0
    else
      status=$?
    fi
    [ "$status" -eq 0 ] && break
    log_disk "failed ${target} attempt ${attempt}"
    prune_if_low
    if [ "$attempt" -ge "$docker_build_attempts" ]; then
      exit "$status"
    fi
    attempt=$((attempt + 1))
    sleep 15
  done
  echo "Pushed ${image}"
  prune_if_low
}

install_docker
clone_repo

echo "$DOCKERHUB_TOKEN" | docker login -u "$DOCKERHUB_USERNAME" --password-stdin
docker buildx create --name enhancerbuilder --use 2>/dev/null || docker buildx use enhancerbuilder

targets_csv="${BUILD_TARGETS:-}"
if [ -z "${targets_csv// }" ]; then targets_csv="$BUILD_TARGET"; fi
EXPANDED_TARGETS=()
IFS=',' read -ra targets <<< "$targets_csv"
for raw in "${targets[@]}"; do
  target="$(echo "$raw" | xargs)"
  [ -z "$target" ] && continue
  case "$target" in
    all) append_all_targets ;;
    remaining) append_remaining_targets ;;
    fast) append_fast_targets ;;
    quality) append_quality_targets ;;
    *) append_target "$target" ;;
  esac
done

if [ "${#EXPANDED_TARGETS[@]}" -eq 0 ]; then
  echo "No enhancer targets to build after expansion."
  exit 0
fi

echo "Expanded enhancer targets: ${EXPANDED_TARGETS[*]}"
for target in "${EXPANDED_TARGETS[@]}"; do
  build_one_target "$target"
done
