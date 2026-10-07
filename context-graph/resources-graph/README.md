# resources-graph

Remembers the public GitHub issues, pull requests and repositories an agent explored, as shared Resources in [Memgraph](https://memgraph.com), and serves them back from memory so the agent doesn't fetch them again.

> Part of the [Context Graph](../README.md) family — wired into `agent-context-graph` as the `resources-graph` connector. Vocabulary: [`CONTEXT.md`](CONTEXT.md). Design: [PRD #464](https://github.com/memgraph/ai-toolkit/issues/464).

## How it works

1. **Touch** — when the agent fetches GitHub itself (WebFetch, `gh issue|pr|repo view`, `gh api repos/...`, `curl` to `api.github.com`, a GitHub MCP tool) or a prompt mentions a GitHub link, the hook records an address-only Touch (FETCHED or PROMPTED). Hooks never call GitHub.
2. **Sweep** — `resources-graph sweep` (out of band: after a session, or on a schedule) fetches what pending Touches name, at full depth and public-only, and stores it as Resources. What it can't fetch stays as an Unresolved Touch with a reason (`not_found`, `not_public`, `rate_limited`).
3. **Cache Read** — the model's `resource` tool (served by `agent-context-graph mcp`) returns a stored Resource with `fetched_at` and GitHub's `updated_at`; the model decides whether that is fresh enough. It never calls GitHub. Each call is recorded as a Touch with outcome `hit` or `miss`.

Resources belong to no user and are shared; a Touch is private to the user whose session made it.

## Graph model

```
(:Session)<-[:IN_SESSION]-(:Touch)-[:TOUCHED]->(:Resource)
(:Resource:Repository)-[:HAS_ISSUE|HAS_PULL_REQUEST]->(:Resource:Issue|PullRequest)
(:Resource:Issue|PullRequest)-[:HAS_COMMENT]->(:Comment)
```

- **Resource** — keyed on GitHub's `node_id`; `address` (`github:owner/repo#123`) is a mutable lookup, `fetched_at` and `updated_at` are on every Resource. Issue/PR: title, body, state, author, labels, assignees, milestone, timestamps, comment count; PR adds merged, base ref and changed-file paths (no diff). Repository: description, README, stars, `pushed_at`.
- **Comment** — one per comment, review comments with their file `path`; captured with its item, never touched on its own.
- **Touch** — `address`, `provenance` (FETCHED / PROMPTED), `served_from_memory`, `outcome` (Cache Reads), `status` (pending / resolved / unresolved) and `reason`, `tool_use_id`, `agent_name`, `at`.

## Setup

```bash
pip install 'resources-graph[agent-context-graph]'
agent-context-graph config set github.token   # prompts; any token that can read public repos
resources-graph setup                          # indexes and constraints
```

Enable the connector in your hooks with `--connector resources-graph`, then sweep whenever you like:

```bash
resources-graph sweep
```

## Not yet

Listings (`gh issue list`), links between Resources and to the Actions that caused a Touch, refresh by conditional requests, a token-less path, and the pre-fetch nudge are the next slices of [PRD #464](https://github.com/memgraph/ai-toolkit/issues/464).
