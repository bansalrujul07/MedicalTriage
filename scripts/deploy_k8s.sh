#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   IMAGE=<registry/image:tag> ./scripts/deploy_k8s.sh

IMAGE="${IMAGE:-medicaltriage:latest}"
DEPLOYMENT_FILE="deployment/k8s/deployment.yaml"
SERVICE_FILE="deployment/k8s/service.yaml"

if ! command -v kubectl >/dev/null 2>&1; then
  echo "Error: kubectl not found."
  exit 1
fi

kubectl apply -f "$SERVICE_FILE"
kubectl apply -f "$DEPLOYMENT_FILE"

kubectl set image deployment/medicaltriage-api api="$IMAGE" --record
kubectl rollout status deployment/medicaltriage-api

echo "Deployment updated to image: $IMAGE"
