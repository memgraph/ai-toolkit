---
description: Run sql2graph in a background Docker container and exec into it
user-invocable: true
---

Start the container in background:
```bash
docker run -d --rm --net memgql-net --name sql2graph-dev --env-file .env -v $(pwd)/output:/output --entrypoint sleep sql2graph infinity
```

Exec into it:
```bash
docker exec -it sql2graph-dev uv run main.py --mapping /output/mapping.json
```
