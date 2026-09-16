# Config file as the sole runtime source for hook subprocesses

Agent runtimes (Claude Code, Codex) spawn hook commands as non-interactive subprocesses that do not source shell profile files (`~/.zshrc`, `~/.bashrc`). Environment variables set only in those profiles never reach hook processes. Rather than document fragile workarounds (e.g. "put exports in `~/.zshenv`"), we decided that hook subprocesses resolve configuration exclusively from a persistent config file at `~/.config/context-graph/config.toml` (after CLI flags). Environment variables are not consulted at hook runtime — they serve only as a write-time convenience during `bootstrap` or `config set`, which persist their values to the config file.

This avoids drift between what `doctor` sees (interactive shell, env vars present) and what hooks see (subprocess, env vars absent), and gives users a single canonical way to configure hooks: the config file.

**Amended by** ADR 0003 (an env var may select *which* config file is read, never what's in it) and ADR 0005 (the config file also covers LLM API keys, and their resolved values are passed as an explicit `env=` to a further subprocess a hook spawns). Both amendments preserve this ADR's core guarantee: configuration *values* still come only from the file.
