# Configuration and capture privacy

[Back to Memgraph Context Graph](../README.md)

## Configuration

`setup` and the `config` command write `~/.config/context-graph/config.toml` (mode `0600`).
The file lists every key, each under a one-line comment saying what it does; a key you haven't set
is written commented out, under a line saying what unset means, so uncommenting it is all it takes
to change it. `setup` and `config set` rewrite the file: your values are kept, comments you add are not.
A `[recall]` or `[extraction]` value is checked by sessions-graph when you set it, so a value it would
reject never reaches the file.
One table, `memgraph_context_graph/config_schema.py`, defines every key: the file, `config show`,
`config get` and `config set` all follow it. **Hook subprocesses resolve their configuration from CLI flags and this file only — never from environment variables** (they don't inherit your shell), per [ADR 0002](adr/0002-config-file-only-hook-resolution.md).
Neither does setup: it writes only `memgraph.url` and `identity.user_id`, from its flags or what it
found. Everything else is set with `config set`, and setup leaves it alone when it rewrites the file.

```toml
[identity]
user_id = "your-name"

[memgraph]
url = "bolt://localhost:7687"
user = ""
password = ""
database = "memgraph"

[llm]
openai_api_key = ""
anthropic_api_key = ""
derive_model = "claude-sonnet-5-5"

[capture]
connectors = "skills-graph, actions-graph, sessions-graph"
exclude_projects = "~/clients/*, ~/personal"
exclude_tools = "TodoWrite, mcp__vault__*"

[redact]
ticket = 'ACME-[0-9]+'

[reconcile]
auto_reconcile = true

[recall]
embedding_model = "BAAI/bge-small-en-v1.5"
turns_k = "8"

[extraction]
model = "fastino/gliner2.5-base-v1"
entity_threshold = "0.3"
```

`[reconcile]` turns on sessions-graph's auto-trigger reconciliation (see
[sessions-graph § reconciliation](../../sessions-graph/README.md#session-reconciliation)),
which needs no LLM. `[llm]` is optional: a key only lets reconciliation learn
the user's own ontology (`sessions-graph derive`), and `derive_model` picks
the model it calls (default `claude-sonnet-5-5` with Anthropic, `gpt-5` with OpenAI).
`[capture] connectors` is which graph connectors a hook enables when its
command names no `--connector`, from `skills-graph`, `actions-graph`,
`sessions-graph` and `resources-graph`; unset means the first three, and
an empty value enables none. Setup installs the ones it names.
The rest of `[capture]` and `[redact]` are the capture policy, applied to every
event before any connector stores it (see [Capture privacy](#capture-privacy)).
A commented-out key means "never configured," distinct from an explicit
`auto_reconcile = false`.
`[recall]` is optional too: without it, recall runs with the widths the
benchmark measured; see [Recall](memory.md#recall-memory-for-the-harnesss-model).
`[ontology]` is optional as well: `path` names a schema file the user's
sessions are extracted under and `derive` is `extend` (the default) or
`off`; see sessions-graph's [Ontology versions](../../sessions-graph/docs/reconciliation.md#ontology-versions).
`[extraction]` tunes how reconciliation, re-extraction and derivation read
sessions, all optional: `model` (a GLiNER2 checkpoint), `entity_threshold` and
`relation_threshold` (drop extractions below that confidence, 0 to 1), and
`chunk_size`/`chunk_overlap` (words per window); unset keys keep GLiNER2's defaults.

Manage it with:

```bash
memgraph-context-graph config show
memgraph-context-graph config get memgraph.url
memgraph-context-graph config set <key> <value>
memgraph-context-graph config set recall.turns_k 12
memgraph-context-graph config set recall.turns_k ""   # back to the default
memgraph-context-graph config --help                  # every key, with what it does
# [redact] patterns are named by you, so they are edited in the file.
```

## Capture privacy

Every hook event passes the capture policy before any connector sees it, so
what it removes never reaches Memgraph:

- **`<private>…</private>`** is stripped from prompts, replies and tool input
  and output. An unclosed `<private>` hides the rest of that text, and a prompt
  that was nothing but private isn't stored at all.
- **Secrets are redacted** to `[REDACTED:<name>]`: AWS, GitHub, Anthropic,
  OpenAI, Slack, Stripe and Google keys, JWTs, PEM private keys, passwords in
  URLs, and the credentials in this config file. Each `[redact]` key adds a
  pattern (a Python regular expression; quote it with single quotes).
  `capture.redact = false` turns redaction off; `<private>` still applies.
- **Long text is clipped** at `capture.max_chars` characters per field
  (default 100000; `0` for no limit).
- **`capture.exclude_tools`** drops those tools' calls and results, by exact
  name or glob (`mcp__vault__*`).
- **`capture.exclude_projects`** captures nothing from a session whose working
  directory is one of these paths or globs, or inside one (`~` expands). The
  agent's own hooks still answer normally, and recall still works there.

Redaction is pattern-based: it catches known credential shapes, not every
secret. What a session stored before the policy existed stays as it was.
Library callers driving `AgentLink` directly pass a `CapturePolicy` themselves.

## Forgetting sessions

```bash
memgraph-context-graph forget session <session-id> [<session-id> ...]
memgraph-context-graph forget project ~/work/client-repo   # every session whose working directory is in it
memgraph-context-graph forget project . --dry-run          # list what matches, delete nothing
```

`forget` lists the matching sessions and asks before deleting (`--yes` skips the
question; without a terminal it is required). For each session it deletes, for
good and with no archive:

- what was captured: its Actions, the Agents only it used, its skill uses and
  resource Touches;
- what was derived from it: its Chunks, entity mentions and extracted
  relationships, any entity left with no other evidence, and the `anon-` User
  made up for it;
- its lines in the local capture-failure log.

What anything else still backs stays: entities other sessions mention, Skills,
public Resources, and your memory files, including ones the session wrote (their
versions just stop naming it). A forgotten session leaves only a
`(:ForgottenSession {session_id})` marker, which stops a hook still running, a
reconciliation in progress, or a later resume of the same session from capturing
it again. It runs against your own graph; in Enterprise, shared spaces are not
touched.
