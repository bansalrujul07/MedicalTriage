#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   GHCR_USERNAME=<github_user_or_org> GHCR_TOKEN=<token> ./scripts/deploy_ghcr.sh [tag]

TAG="${1:-latest}"
IMAGE_NAME="medicaltriage"
GHCR_USERNAME="${GHCR_USERNAME:-}"
GHCR_TOKEN="${GHCR_TOKEN:-}"

if [[ -z "$GHCR_USERNAME" || -z "$GHCR_TOKEN" ]]; then
  echo "Error: GHCR_USERNAME and GHCR_TOKEN are required."
  exit 1
fi

FULL_IMAGE="ghcr.io/${GHCR_USERNAME}/${IMAGE_NAME}:${TAG}"

echo "$GHCR_TOKEN" | docker login ghcr.io -u "$GHCR_USERNAME" --password-stdin

docker build -t "$FULL_IMAGE" .
docker push "$FULL_IMAGE"

echo "Pushed: $FULL_IMAGE"
