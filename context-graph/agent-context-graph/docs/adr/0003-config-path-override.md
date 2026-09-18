# An environment variable may select the config file, never its contents

ADR 0002 made config file sole runtime source for hook subprocesses; env vars not consulted at hook runtime. Still true of config *values*. This ADR carves one exception: `CONTEXT_GRAPH_CONFIG` selects **which file** is read.

## Why the exception is needed

Config path was single global (`~/.config/context-graph/config.toml`), no override. Hooks resolve everything from it -> pointing one Claude Code session at a different Memgraph meant rewriting that file, redirecting **every** session on the machine, not just the intended one.

Not hypothetical: found while building eval gold slice (drives real session against dedicated eval instance). With global config temporarily repointed, an unrelated Claude Code session's activity got recorded into the eval graph — exactly the ambient-session pollution that decision was meant to prevent. Mechanism defeated its own purpose.

No way to isolate a session short of changing `HOME` or running in a container.

## Why it doesn't contradict ADR 0002

ADR 0002 was about **ambient** environment: hook subprocesses don't source shell profiles, so `~/.zshrc` values never reached them, and `doctor` (interactive, env present) disagreed with hooks (subprocess, env absent). That ADR removed config depending on where a process happened to launch from.

A config *path* handed down by the process spawning the session is the opposite: explicit, set by parent for its own child, drift-free — nothing ambient to drift from. `doctor` and the hooks it inspects agree by passing the same variable.

Distinction: **which file** vs **what's in it**. Values still come only from the file — ADR 0002's guarantee intact.

## Consequences

- Callers needing an isolated session write their own config file, pass its path. `context_graph_eval.live.hooks_pointed_at` does this.
- Unset (normal case): behavior unchanged.
- `config set`/`bootstrap` write to the overridden path when set — makes a temporary config usable.
- `doctor` inspects whichever file the variable names — a session under an override diagnosable with same tool.
