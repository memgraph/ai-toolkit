# Config file also covers LLM API keys; explicit env for spawned subprocesses

ADR 0002 made `~/.config/context-graph/config.toml` the sole runtime source
for hook subprocess configuration, covering identity and Memgraph connection
settings. That resolution happens as constructor kwargs passed directly to
`SkillGraph`/`SessionsGraph`/`ActionsGraph` inside the hook process — it is
never written back into that process's `os.environ`.

`SessionsGraphConnector` (sessions-graph) spawns a **further** detached
subprocess on `SESSION_END` when `auto_reconcile` is enabled: `sessions-graph
reconcile --session <id>`, which does LLM-backed entity extraction via
LightRAG. Being a plain `subprocess.Popen(...)` with no explicit `env=`, that
child only inherited whatever ambient `os.environ` the parent hook process
happened to have — which, per ADR 0002, does *not* include the resolved
Memgraph config, and never had a path to an LLM API key at all (LightRAG's
default `llm_model_func` reads `OPENAI_API_KEY` straight from `os.environ`
and raises if unset).

We decided to:

1. Extend `HookConfig`/`config.toml` with an `[llm]` section
   (`openai_api_key`, `anthropic_api_key`), resolved via `resolve_llm_env()`,
   mirroring `resolve_memgraph_env()`. `bootstrap` captures these from
   `OPENAI_API_KEY`/`ANTHROPIC_API_KEY` in the environment at write time, same
   as it already does for Memgraph credentials — env vars remain a write-time
   convenience only, never consulted at hook runtime.
2. Have the spawn site (`SessionsGraphConnector._spawn_reconciliation`) build
   an **explicit `env=`** for the child: a copy of the current `os.environ`
   overlaid with non-empty values from `resolve_memgraph_env()` and
   `resolve_llm_env()`. This guarantees the detached subprocess gets what the
   hook process resolved, regardless of what the harness's own ambient
   environment contained.
3. As defense in depth, `sessions-graph reconcile` (run standalone via cron or
   manually, not only spawned by the hook) also best-effort fills in the same
   config-file values via `os.environ.setdefault` at startup, so it behaves
   consistently whether invoked by the hook or by hand. This uses an optional
   import of `agent_context_graph` and is a no-op if that package isn't
   installed, since `sessions-graph`'s `reconciliation` extra doesn't require
   it.

This keeps the same philosophy as ADR 0002 (config file is canonical) while
closing the gap it didn't cover: subprocesses spawned *by* a hook process,
not just the hook process itself.
