---
description: Build and publish the structured2graph Docker image to Docker Hub
user-invocable: true
---

Read the current version from `pyproject.toml` (the `version` field under `[project]`).

Build and push with both the version tag and `latest`:
```bash
docker build -t memgraph/structured2graph:<version> -t memgraph/structured2graph:latest .
docker push memgraph/structured2graph:<version>
docker push memgraph/structured2graph:latest
```

Replace `<version>` with the actual version from `pyproject.toml`.
