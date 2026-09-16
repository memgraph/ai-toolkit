# An environment variable may select the config file, never its contents

ADR 0002 made the config file the sole runtime source for hook subprocesses and
stated that environment variables are not consulted at hook runtime. That
remains true of configuration *values*. This ADR carves out one exception:
`CONTEXT_GRAPH_CONFIG` selects **which file** is read.

## Why the exception is needed

The config path was a single global, `~/.config/context-graph/config.toml`, with
no override. Since hooks resolve everything from it, pointing one Claude Code
session at a different Memgraph meant rewriting that file — which redirects
**every** session running on the machine, not just the intended one.

This is not hypothetical. It was found while building the eval gold slice, which
drives a real session against a dedicated eval instance. With the global config
temporarily repointed, an unrelated Claude Code session's activity was recorded
into the eval graph. An eval graph contaminated by ambient sessions is exactly
the pollution that decision was designed to prevent, so the mechanism defeated
its own purpose.

There was no way to isolate a session short of changing `HOME` or running in a
container.

## Why it does not contradict ADR 0002

ADR 0002's reasoning was about **ambient** environment: hook subprocesses do not
source shell profiles, so values set in `~/.zshrc` never reached them, and
`doctor` (interactive, env present) disagreed with hooks (subprocess, env
absent). Configuration that depends on where a process happened to be launched
from is what that ADR removed.

A config *path* handed down by the process that spawns the session is the
opposite case: explicit, set by the parent for its own child, and drift-free
because there is nothing ambient to drift from. `doctor` and the hooks it
inspects can be made to agree by passing the same variable.

The distinction that matters is **which file** versus **what is in it**. Values
still come only from the file, so ADR 0002's guarantee is intact.

## Consequences

- Callers that need an isolated session write their own config file and pass its
  path. `context_graph_eval.live.hooks_pointed_at` does this.
- Unset — the normal case — behaviour is exactly as before.
- `config set` and `bootstrap` write to the overridden path when one is set,
  which is what makes a temporary config usable at all.
- `doctor` inspects whichever file the variable names, so a session under an
  override can be diagnosed with the same tool.
