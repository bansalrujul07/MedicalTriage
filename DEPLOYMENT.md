# Deployment Guide

## Prerequisites
- Docker installed and running
- Optional: kubectl configured for your cluster
- Repository root contains `Dockerfile`

## 1) Local Run
```bash
docker build -t medicaltriage:latest .
docker run --rm -p 8000:8000 --env-file .env medicaltriage:latest
```

Health check:
```bash
curl -fsS http://127.0.0.1:8000/health
```

## 2) Docker Compose
```bash
docker compose up --build -d
```

## 3) Push to Docker Hub
Set credentials:
```bash
export DOCKERHUB_USERNAME=<your-user>
export DOCKERHUB_TOKEN=<your-token>
```

Push image:
```bash
./scripts/deploy_dockerhub.sh latest
```

## 4) Push to GitHub Container Registry (GHCR)
Set credentials:
```bash
export GHCR_USERNAME=<github-user-or-org>
export GHCR_TOKEN=<github-token-with-package-write>
```

Push image:
```bash
./scripts/deploy_ghcr.sh latest
```

## 5) Deploy to Kubernetes
Apply manifests and set image:
```bash
IMAGE=<registry/image:tag> ./scripts/deploy_k8s.sh
```

Default manifests:
- `deployment/k8s/deployment.yaml`
- `deployment/k8s/service.yaml`

## 6) Manual Readiness Check
Run local tests and image build before release:
```bash
python -m pytest -q
docker build -t medicaltriage:ci .
```
