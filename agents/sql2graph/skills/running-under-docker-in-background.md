---
description: Run structured2graph in a background Docker container and exec into it
user-invocable: true
---

Build the image (use Dockerfile.local for local source):
```bash
docker build -f Dockerfile.local -t memgraph/structured2graph .
```

Start the container in background:
```bash
docker run -d --rm --net memgql-net --name structured2graph-dev --env-file .env -v $(pwd)/output:/output --entrypoint sleep memgraph/structured2graph infinity
```

Exec into it:
```bash
docker exec -it structured2graph-dev uv run main.py --mapping /output/mapping.json
```
