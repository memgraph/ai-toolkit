---
description: Build and publish the structured2graph Docker image to Docker Hub
user-invocable: true
---

Read the current version from `pyproject.toml` (the `version` field under `[project]`).

Build and push multi-platform (amd64 + arm64) with both the version tag and `latest`:
```bash
docker buildx build --platform linux/amd64,linux/arm64 -t memgraph/structured2graph:<version> -t memgraph/structured2graph:latest --push .
```

Replace `<version>` with the actual version from `pyproject.toml`.
