# A process a hook spawns reads the config file itself; only the file's path is handed down

ADR 0005 closed a gap ADR 0002 missed: the detached `sessions-graph` process a hook starts never saw the config file's values. It did that by copying them. The parent built the child's `env=` from its own resolved Memgraph and LLM values, and `sessions-graph`'s CLI also filled unset variables from the file at startup "as defence in depth". This ADR supersedes ADR 0005's point 2 and makes its point 3 the only mechanism.

## Context

Copying values meant two code paths, and they disagreed about which value wins. The parent's copy overwrote the environment, so the file won. The child's fill used `setdefault`, so the environment won. A third copy in `context-graph-eval` followed the child's rule. Each new config-backed variable had to be added to every copy. Architecture review #549 (item 4) flagged this as the ambient drift ADR 0002 exists to prevent.

It also put credentials in the child's environment, where anything that dumps the environment can see them.

## Decision

- **The parent passes the path, not the values.** `_identity.spawn_env()` returns the hook's environment minus every config-backed variable (`MEMGRAPH_URL/USER/PASSWORD/DATABASE`, the provider keys, and LightRAG's `MEMGRAPH_URI`/`MEMGRAPH_USERNAME` aliases). It adds `CONTEXT_GRAPH_CONFIG` naming the file the hook read. This is ADR 0003's rule: a parent selects which file, never what is in it.
- **The child reads the file itself.** `_identity.fill_env_from_config()` is the one place config values become environment variables. A variable already set wins. In a hook-spawned child none is set, so the file decides, and an empty key in the file can't be filled by an ambient credential. On a hand run, `MEMGRAPH_URL=… sessions-graph reconcile` still targets that instance, which `scripts/dev-memgraph.sh reconcile` relies on.
- **Empty values are set too.** An Enterprise login's unnamed database is `MEMGRAPH_DATABASE=""`, which memgraph-toolbox reads as "the login's main database", where unset would mean `memgraph`.
- `context-graph-eval` fills only the LLM keys (`memgraph=False`), because it picks its own Memgraph.

## Consequences

- Every config-backed variable is listed once, in `_identity.CONFIG_ENV_KEYS`.
- Embed children now read the provider keys from the file too. ADR 0005 left them out because embedding runs inside Memgraph. They come from the same file the hook already trusts, and nothing in the embed path reads them.
- A spawned process must be able to read the file the hook read. It runs as the same user, from the same path, a moment later. Editing the file between the two is the same race as editing it between two hooks.
