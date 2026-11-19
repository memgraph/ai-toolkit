---
description: Build the release Docker image locally and/or publish it to Docker Hub
user-invocable: true
---

Read the current version from `pyproject.toml` (the `version` field under `[project]`). Before building, check with the user if that version is correct — they may want to bump it first (e.g. the pinned version in `Dockerfile` was changed manually).

## Build locally

Build the release image using the production `Dockerfile`:
```bash
docker build -f Dockerfile.local -t memgraph/structured2graph:<version> -t memgraph/structured2graph:latest .
```

This is sufficient when only the Docker setup changed (e.g. bumped the pinned PyPI version, changed system packages, or updated the base image) and you want to test locally before publishing.

## Publish to Docker Hub

Build and push multi-platform (amd64 + arm64) with both the version tag and `latest`:
```bash
docker buildx build --platform linux/amd64,linux/arm64 -t memgraph/structured2graph:<version> -t memgraph/structured2graph:latest --push .
```

Replace `<version>` with the confirmed version.
