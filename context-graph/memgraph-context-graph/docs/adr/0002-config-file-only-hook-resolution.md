# Config file as the sole runtime source for hook subprocesses

Agent runtimes (Claude Code, Codex) spawn hook commands as non-interactive subprocesses — don't source shell profile files (`~/.zshrc`, `~/.bashrc`). Env vars set only there never reach hook processes. Instead of documenting fragile workarounds ("put exports in `~/.zshenv`"): hook subprocesses resolve config exclusively from a persistent file at `~/.config/context-graph/config.toml` (after CLI flags). Env vars not consulted at hook runtime — only a write-time convenience during `bootstrap`/`config set`, which persist values to the config file.

Avoids drift between what `doctor` sees (interactive shell, env vars present) and what hooks see (subprocess, env vars absent). One canonical way to configure hooks: the config file.

**Amended by** ADR 0003 (env var may select *which* config file is read, never what's in it), ADR 0005 (config file also covers LLM API keys) and ADR 0007 (a subprocess a hook spawns is handed the config file's path and reads it itself). All preserve this ADR's core guarantee: config *values* still come only from the file.

**Since superseded detail:** env vars are no longer a write-time input either. `bootstrap` is gone, and `setup` and `config set` take values from the user, so nothing copies the environment into the file.
