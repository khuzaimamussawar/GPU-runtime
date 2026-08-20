#!/usr/bin/env bash
set -Eeuo pipefail
: "${BUILD_TARGET:?BUILD_TARGET fast|quality is required}"
: "${IMAGE_TAG:?IMAGE_TAG is required}"
: "${DOCKERHUB_USERNAME:?DOCKERHUB_USERNAME is required}"
: "${DOCKERHUB_TOKEN:?DOCKERHUB_TOKEN is required}"
: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"
: "${GITHUB_REF_NAME:?GITHUB_REF_NAME is required}"
repo_dir=/opt/scenebuilder-gpu-runtime-enhancer
install_docker(){ command -v docker >/dev/null 2>&1 && return; apt-get update; apt-get install -y ca-certificates curl git; curl -fsSL https://get.docker.com | sh; }
install_docker
rm -rf "$repo_dir"
if [ -n "${GH_FH_TOKEN_MM_H3_SERVERLESS:-}" ]; then
  git clone --depth 1 --branch "$GITHUB_REF_NAME" "https://x-access-token:${GH_FH_TOKEN_MM_H3_SERVERLESS}@github.com/${GITHUB_REPOSITORY}.git" "$repo_dir"
else
  git clone --depth 1 --branch "$GITHUB_REF_NAME" "https://github.com/${GITHUB_REPOSITORY}.git" "$repo_dir"
fi
echo "$DOCKERHUB_TOKEN" | docker login -u "$DOCKERHUB_USERNAME" --password-stdin
docker buildx create --name enhancerbuilder --use 2>/dev/null || docker buildx use enhancerbuilder
case "$BUILD_TARGET" in
  fast) image="${DOCKERHUB_USERNAME}/scenebuilder-enhancer-fast:${IMAGE_TAG}"; dockerfile="enhancer/docker/Dockerfile.fast" ;;
  quality) image="${DOCKERHUB_USERNAME}/scenebuilder-enhancer-quality:${IMAGE_TAG}"; dockerfile="enhancer/docker/Dockerfile.quality" ;;
  *) echo "Unknown enhancer target: $BUILD_TARGET" >&2; exit 2 ;;
esac
cd "$repo_dir"
docker buildx build --file "$dockerfile" --tag "$image" --push --progress plain .
echo "Pushed $image"
