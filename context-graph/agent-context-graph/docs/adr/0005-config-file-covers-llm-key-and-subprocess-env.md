# Config file also covers LLM API keys; explicit env for spawned subprocesses

ADR 0002 made `~/.config/context-graph/config.toml` sole runtime source for hook subprocess config: identity + Memgraph connection settings. Resolution happens as constructor kwargs passed directly to `SkillGraph`/`SessionsGraph`/`ActionsGraph` inside hook process — never written back to that process's `os.environ`.

`SessionsGraphConnector` (sessions-graph) spawns a **further** detached subprocess on `SESSION_END` when `auto_reconcile` enabled: `sessions-graph reconcile --session <id>`, LLM-backed entity extraction via LightRAG. Plain `subprocess.Popen(...)`, no explicit `env=` -> child only inherited whatever ambient `os.environ` parent hook process had — which, per ADR 0002, excludes resolved Memgraph config, and never had a path to an LLM API key at all (LightRAG's default `llm_model_func` reads `OPENAI_API_KEY` straight from `os.environ`, raises if unset).

Decided:

1. Extend `HookConfig`/`config.toml` with `[llm]` section (`openai_api_key`, `anthropic_api_key`), resolved via `resolve_llm_env()`, mirroring `resolve_memgraph_env()`. `bootstrap` captures these from `OPENAI_API_KEY`/`ANTHROPIC_API_KEY` in env at write time, same as Memgraph credentials — env vars stay write-time convenience only, never consulted at hook runtime.
2. Spawn site (`SessionsGraphConnector._spawn_reconciliation`) builds **explicit `env=`** for child: copy of current `os.environ` overlaid with non-empty `resolve_memgraph_env()`/`resolve_llm_env()` values. Guarantees detached subprocess gets what hook process resolved, regardless of harness's own ambient environment.
3. Defense in depth: `sessions-graph reconcile` (standalone via cron/manual, not only hook-spawned) also best-effort fills same config-file values via `os.environ.setdefault` at startup — consistent whether invoked by hook or by hand. Optional import of `agent_context_graph`, no-op if not installed (sessions-graph's `reconciliation` extra doesn't require it).

Same philosophy as ADR 0002 (config file canonical), closes the gap it missed: subprocesses spawned *by* a hook process, not just the hook process itself.
