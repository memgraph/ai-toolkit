# resources-graph

Remembers the public GitHub issues, pull requests, repositories and issue/PR lists an agent explored, as shared Resources in [Memgraph](https://memgraph.com), and serves them back from memory so the agent doesn't fetch them again.

> Part of the [Context Graph](../README.md) family — wired into `agent-context-graph` as the `resources-graph` connector. Vocabulary: [`CONTEXT.md`](CONTEXT.md). Design: [PRD #464](https://github.com/memgraph/ai-toolkit/issues/464).

## How it works

1. **Touch** — when the agent fetches GitHub itself (WebFetch, `gh issue|pr|repo view`, `gh issue|pr list`, `gh api repos/...`, `curl` to `api.github.com`, a GitHub MCP tool) or a prompt mentions a GitHub link, the hook records an address-only Touch (FETCHED or PROMPTED). A listing is one Touch, not one per member. Hooks never call GitHub.
2. **Sweep** — `resources-graph sweep` (out of band: after a session, or on a schedule) fetches what pending Touches name, at full depth and public-only, and stores it as Resources; a listing's members are fetched up to the limit the agent asked for. What it can't fetch stays as an Unresolved Touch with a reason (`not_found`, `not_public`, `rate_limited`).
3. **Cache Read** — the model's `resource` tool (served by `agent-context-graph mcp`) returns a stored Resource with `fetched_at` and GitHub's `updated_at`; the model decides whether that is fresh enough. A listing comes back as a paged index; members are read one at a time. It never calls GitHub. Each call is recorded as a Touch with outcome `hit`, `subsumed` or `miss`.
4. **Nudge** — just before the agent fetches something memory already holds, one line tells it so (with the same facts) — on Claude Code and Codex, the harnesses whose pre-tool hook can add context without touching the call. The fetch always runs.

Resources belong to no user and are shared; a Touch is private to the user whose session made it.

## Listings

`gh issue list -R memgraph/memgraph --label bug` becomes one shared **Listing**, keyed by repository, kind and normalised filters (state, labels, author, assignee, milestone) or the exact search text. It is answered from memory when:

- **hit** — the same Listing is stored and holds every member, or at least as many as asked (members keep GitHub's order, newest created first, so a shorter ask is a prefix);
- **subsumed** — a *fully expanded* Listing of the same repository and kind filters at most as narrowly (all open issues answer `--label bug`), and the stored members are filtered locally.

Free-text searches only hit exactly. A truncated Listing (`--limit 500` of 686) answers only shorter asks of itself.

## Freshness

No TTL and no background crawl: a Resource is revalidated only when the agent really fetches it again (the model judged memory too stale). Then the Sweep makes one cheap check of GitHub's `updatedAt` — for a Listing, a light index of its members — and refetches at full depth only what changed; unchanged content just gets a new `fetched_at`. Reactions don't move GitHub's `updatedAt`, so reaction counts may be stale.

Closed `wayfinder:map` issues are served with a reading note: their build plans may
already have shipped. A recent fetch does not make the source narrative current;
check linked implementation PRs, current code and releases before proposing pending work.

When GitHub rate-limits, the Sweep stops; that Touch is kept as `rate_limited` and it and the rest are retried by the next Sweep.

A transferred issue gets a new GitHub id (the id embeds its repository). When a fetch finds an item somewhere new, the old Resource releases its Address and points at the new one with `MOVED_TO`.

## Graph model

```
(:Session)<-[:IN_SESSION]-(:Touch)-[:TOUCHED]->(:Resource | :Listing)
(:Resource:Repository)-[:HAS_ISSUE|HAS_PULL_REQUEST]->(:Resource:Issue|PullRequest)
(:Resource:Issue|PullRequest)-[:HAS_COMMENT]->(:Comment)
(:Listing)-[:LISTS_FROM]->(:Repository), (:Listing)-[:HAS_MEMBER {position}]->(:Issue|PullRequest)
(:Resource)-[:REFERENCES]->(:Resource), (:PullRequest)-[:CLOSES]->(:Issue)
(:Resource)-[:MOVED_TO]->(:Resource)
(:Touch)-[:CAUSED_BY]->(:ToolCall), (:Touch)-[:BY_AGENT]->(:Agent)   -- actions-graph's nodes
```

- **Resource** — keyed on GitHub's `node_id`; `address` (`github:owner/repo#123`) is a mutable lookup, `fetched_at` and `updated_at` are on every Resource. Issue/PR: title, body, state, author, labels, assignees, milestone, timestamps, comment count; PR adds merged, base ref and changed-file paths (no diff). Repository: description, README, stars, `pushed_at`.
- **Comment** — one per comment, review comments with their file `path`; captured with its item, never touched on its own.
- **Listing** — shared like a Resource but not one: `key`, `limit`, `total_count`, `member_count`, `fully_expanded`, `fetched_at`; a refetch replaces its members.
- **REFERENCES / CLOSES** — drawn only between stored Resources, whichever end was stored first; references to anything not in memory are kept on the item as URLs (`referenced_by_urls`, `closes_urls`).
- **Touch** — `address`, `provenance` (FETCHED / PROMPTED), `served_from_memory`, `outcome` (Cache Reads), `status` (pending / resolved / unresolved) and `reason`, `tool_use_id`, `agent_name`, `at`. The Sweep links it to the actions-graph `ToolCall` (same `tool_use_id`, same Session) and subagent `Agent` it came from, when actions-graph recorded them.

## Setup

```bash
pip install 'resources-graph[agent-context-graph]'
resources-graph setup   # indexes and constraints
```

The Sweep fetches with your existing `gh` login (`gh auth login`) — the same access your agent reads GitHub with. GitHub's GraphQL API needs a token even for public data. On a machine without `gh`, set one instead: `agent-context-graph config set github.token` (it prompts; it also wins over the `gh` login when set). With no credentials at all the Sweep still fetches single issues, PRs and repositories over REST (60 requests an hour); listings wait until there are credentials.

Enable the connector in your hooks with `--connector resources-graph` (add `--connector actions-graph` too for Touch → ToolCall links), then sweep whenever you like:

```bash
resources-graph sweep
```

## Which harnesses show the Nudge

| Harness | Nudge before a fetch | Otherwise |
|---|---|---|
| Claude Code, Codex | yes — `PreToolUse` `additionalContext`, no permission decision | |
| Copilot CLI, Cursor, OpenCode, Grok Build, Antigravity CLI | no — their pre-tool hooks can only allow, deny or rewrite the call | the session-start hint about the `resource` tool |
