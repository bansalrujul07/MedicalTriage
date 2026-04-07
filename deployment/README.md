# Deployment Structure

This folder contains Kubernetes-ready deployment manifests.

## Files
- `k8s/deployment.yaml`: API deployment with readiness/liveness probes
- `k8s/service.yaml`: ClusterIP service exposing HTTP

## Container source
The repository root `Dockerfile` is the default production image build file.

## Quick start
1. Build image:
   docker build -t medicaltriage:latest .
2. Apply manifests:
   kubectl apply -f deployment/k8s/deployment.yaml
   kubectl apply -f deployment/k8s/service.yaml
3. Verify:
   kubectl get pods
   kubectl get svc
