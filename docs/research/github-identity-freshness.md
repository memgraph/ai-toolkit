# GitHub identity and freshness signals for public resources

Research for #456 (map #454, Resources component). Question: what identity and
freshness signals does GitHub give for public repos, issues, PRs and comments,
and how do established GitHub-to-graph ingestion tools model and incrementally
sync GitHub?

Sources are GitHub's own docs (linked inline) plus live probes against
`api.github.com` run on 2026-10-07, marked **[probe]**. Probes used
`memgraph/memgraph` (1,510 issues + 3,465 PRs, 683 open issues at probe time).

## TL;DR

- **Identity:** the stable key is the GraphQL `node_id` (opaque string, typed,
  survives renames/transfers of the repo). `owner/repo#n` and URLs are
  *addresses*, not identities: they change on repo rename/transfer and issue
  transfer, and GitHub only keeps them working via redirects that break if the
  old name is reused. REST numeric `id` is per-type and not unique across types.
- **Issues and PRs share one number space** per repo, and the REST issues
  endpoints return PRs too — `owner/repo#n` alone does not tell you which kind.
- **Freshness:** `updated_at` on an issue/PR bumps when its comments change
  **[probe]**, so it is a cheap "has this thread changed?" signal. List
  endpoints accept `since=<ISO8601>` (filters on `updated_at`). ETags give free
  revalidation — but **only for authenticated requests**: unauthenticated 304s
  still consume the 60/h quota **[probe]**.
- **Cost of a 500-issue sweep:** unauthenticated REST cannot do it within one
  hour's quota (60 req/h; ~5 list pages + ≥87 comment pages for
  memgraph/memgraph). GraphQL is **not available unauthenticated** (403,
  `graphql.limit = 0`) **[probe]**. Authenticated GraphQL fetches 100 issues ×
  up to 100 comments each for **1 point** of 5,000/h **[probe]**, so 500 issues
  with comments ≈ 5 points.
- **Prior art:** inventory-style graph ingesters key nodes on a stable
  identifier (often the canonical URL), stamp every node/relationship touched in
  a run with that run's update tag, then delete in-scope nodes whose tag does not
  match ("mark and sweep"). Incremental ELT-style syncers instead keep an
  `updated_at` cursor per stream, re-read with `since`, dedupe on id, and do
  **not** detect deletions.

## 1. Canonical identity

### Forms available

| Form | Example | Stable across repo rename/transfer? | Notes |
|---|---|---|---|
| Web URL | `https://github.com/memgraph/memgraph/issues/1` | No (redirects) | For a PR, the issues URL redirects to `/pull/1`; `html_url` from the API gives the right one **[probe]** |
| `owner/repo#n` | `memgraph/memgraph#1` | No | Human/agent-facing; what a harness tool call or pasted text usually carries |
| REST `id` | `705520345` (issue id of PR #1) | Yes | Unique **per resource type** only. For a PR fetched via the issues endpoint, `id` is the *issue* id, not the PR id ([issues docs](https://docs.github.com/en/rest/issues/issues)) |
| Repo REST `id` | `297302063` | Yes | API pagination links use `/repositories/297302063/...`, i.e. GitHub itself addresses the repo by id **[probe]** |
| GraphQL `node_id` | `MDExOlB1bGxSZXF1ZXN0NDkwMjMzNjAy` (PR #1) | Yes | Globally unique, encodes the type; returned as `node_id` on every REST object ([global node IDs](https://docs.github.com/en/graphql/guides/using-global-node-ids)) |

GitHub's guidance: "it's best practice to persist the global node ID so you can
easily reference objects across API versions"
([using global node IDs](https://docs.github.com/en/graphql/guides/using-global-node-ids)).
Node IDs come in a legacy format (`MDQ6...`) and a new one (`U_kgDO...`); GitHub
says to treat them as **opaque strings**, can force the new format with
`X-Github-Next-Global-ID: 1`, and warns the legacy format will eventually stop
working ([migrating node IDs](https://docs.github.com/en/graphql/guides/migrating-graphql-global-node-ids)).
So a stored `node_id` may need a one-time rewrite; storing whatever the API
returns *with* that header avoids it.

### Issues and PRs share one number space

"GitHub's REST API considers every pull request an issue, but not every issue
is a pull request"; PRs are distinguished by a `pull_request` key in issue
responses ([issues docs](https://docs.github.com/en/rest/issues/issues)). Probe:
`GET /repos/memgraph/memgraph/issues/1` returns PR #1 — `html_url` is
`.../pull/1` and the `node_id` decodes to type `PullRequest` **[probe]**. The
same is true for comments: the issue-comments endpoints cover PR conversation
comments, while PR *review* comments are a separate API
([issue comments docs](https://docs.github.com/en/rest/issues/comments)).

### Renames and transfers

- **Repo rename:** web traffic and `git` operations to the old name redirect
  ([renaming](https://docs.github.com/en/repositories/creating-and-managing-repositories/renaming-a-repository)).
  Caveat: "do not reuse the original name ... If you do, redirects to the renamed
  repository will no longer work."
- **Repo transfer:** issues, PRs, wiki, stars move with it; "All links to the
  previous repository location are automatically redirected", but redirects are
  "permanently deleted" if a repo/fork is created at the old location
  ([transferring a repo](https://docs.github.com/en/repositories/creating-and-managing-repositories/transferring-a-repository)).
- **REST on a moved repo:** `GET /repos/{owner}/{repo}` documents `301 Moved
  permanently` ([repos docs](https://docs.github.com/en/rest/repos/repos)).
- **Issue transfer:** only between repos of the same owner, never
  private→public; comments and assignees are kept, labels/milestones only if a
  same-named one exists; "The original URL redirects to the new issue's URL"
  ([transferring an issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/administering-issues/transferring-an-issue-to-another-repository)).
  The issue lands under the target repo's numbering, so `owner/repo#n` changes.
  `GET` on the old issue returns **301** ("Issue was permanently moved"), a
  deleted issue returns **410** ([issues docs](https://docs.github.com/en/rest/issues/issues)).
  GitHub's docs do **not** state whether the `node_id` is preserved across an
  issue transfer — unverified here (see open questions).

## 2. Freshness signals

### `updated_at`

Every issue/PR/comment carries `updated_at`. Probe over the 100 most recently
updated memgraph/memgraph issues (47 with comments): no comment had an
`updatedAt` later than its parent issue's `updatedAt` — i.e. comment
creation/edit bumps the issue **[probe]**. So for a cached issue thread,
"issue `updated_at` unchanged" is a sufficient one-request staleness check for
the issue body + its comments. (Not checked: whether reactions or PR review
comments bump it.)

`Last-Modified` is **not** `updated_at`: issue #1 had `updated_at`
2020-09-21 but `Last-Modified: Tue, 01 Sep 2026` **[probe]**. Use the body
field, not the header, for semantic freshness.

### Conditional requests (ETag / If-None-Match)

GitHub: save the `etag`, send it as `if-none-match`; a `304` "does not count
against your primary rate limit" — qualified as requiring the request to be
correctly authorized
([best practices](https://docs.github.com/en/rest/using-the-rest-api/best-practices-for-using-the-rest-api)).
Probe confirms both halves:

| Request | Status | Quota |
|---|---|---|
| Authenticated, `If-None-Match` ×2 | 304 | `X-Ratelimit-Used` unchanged (31 → 31) |
| Unauthenticated, `If-None-Match` ×2 | 304 | consumed one request each (56 → 55) |

ETags are weak (`W/"..."`) and differ between the authenticated and
unauthenticated representation of the same issue (`Vary: Authorization`)
**[probe]** — an ETag cached under one auth mode is useless under the other.
List pages also carry ETags, so "has page 1 of `sort=updated` changed?" can be a
single free authenticated request.

### `since` on list endpoints

- `GET /repos/{o}/{r}/issues?state=all&since=...&sort=updated` — "only items
  updated after" the timestamp; includes PRs; `per_page` ≤ 100
  ([issues docs](https://docs.github.com/en/rest/issues/issues)).
- `GET /repos/{o}/{r}/issues/comments?since=...&sort=updated` — repo-wide
  issue+PR comments; and `.../issues/{n}/comments?since=...` per issue
  ([issue comments docs](https://docs.github.com/en/rest/issues/comments)).
- The repo issues list paginates by cursor (`after=`) with no `rel="last"`;
  the repo comments list has page numbers (87 pages × 100 for memgraph/memgraph)
  **[probe]**.

GitHub also recommends webhooks over polling
([best practices](https://docs.github.com/en/rest/using-the-rest-api/best-practices-for-using-the-rest-api))
— not applicable to a harness reading arbitrary public repos it does not own.

## 3. Rate limits and the 500-issue sweep

| Mode | Limit | Source |
|---|---|---|
| REST unauthenticated | 60 req/h per IP | [REST rate limits](https://docs.github.com/en/rest/using-the-rest-api/rate-limits-for-the-rest-api) |
| REST authenticated user | 5,000 req/h (15,000 Enterprise Cloud) | same |
| `GITHUB_TOKEN` (Actions) | 1,000 req/h per repo | same |
| REST secondary | ≤100 concurrent, ≤900 points/min (GET = 1) | same |
| GraphQL unauthenticated | **not available** — 403, `graphql.limit: 0` | **[probe]** |
| GraphQL authenticated | 5,000 points/h; cost = Σ(connection requests)/100, min 1; `first` ≤ 100; ≤500k nodes/call | [GraphQL limits](https://docs.github.com/en/graphql/overview/rate-limits-and-query-limits-for-the-graphql-api) |

Concrete cost of re-reading ~500 memgraph/memgraph issues with comments:

| Approach | Requests / points | Fits unauthenticated 60/h? |
|---|---|---|
| REST: 5 list pages + 1 comments call per issue | ~505 requests | No (≈8.5 h of quota) |
| REST: 5 list pages + repo-wide comments (all 87 pages) | ~92 requests | No |
| REST incremental: list with `since` + comments with `since` | a few requests when little changed | Yes, if the delta is small |
| REST authenticated revalidation: 500 × `If-None-Match` | 500 requests, **0 quota** if unchanged | n/a (needs auth) |
| GraphQL authenticated: 5 × (100 issues × 100 comments) | **~5 points** of 5,000 (measured 1 point per 100×100 page) | n/a (needs auth) |

So for an **unauthenticated** harness, the bottleneck is not bytes but the
60-request budget, and GitHub gives no free revalidation; the only cheap
freshness path is a `since`-filtered list read. With *any* token, the cost
collapses (free 304s, or ~1 GraphQL point per 100 threads). Note that in
practice the harness, not the Resources component, makes these calls (e.g.
`gh` CLI or a web-fetch tool) — the agent's own auth mode decides which regime
applies.

## 4. How established GitHub-to-graph ingesters model and sync GitHub

Described by process only.

### Inventory / asset-graph ingesters (mark-and-sweep)

- **Model:** one label per GitHub object type (organization, repository, user,
  team, branch, dependency, language, …), each node keyed on a single `id`
  property that is usually the object's **canonical URL** (repo URL, profile
  URL, team URL) or a deterministic composite (repo URL + path). Relationships
  express containment (org → repo), membership/permission (user/team → repo
  with ADMIN/WRITE/READ), and dependency edges. Issue/PR content is generally
  *not* modelled — the goal is inventory, not conversation memory.
- **Sync:** every run picks an update tag (the run's start timestamp). Every
  node and relationship written in that run gets `lastupdated = update_tag` via
  idempotent MERGE.
- **Staleness/cleanup:** after ingest, a cleanup job deletes nodes and
  relationships whose `lastupdated` ≠ the current tag — anything the API no
  longer returned is gone. Cleanup is **scoped** to the sub-resource just synced
  (e.g. only stale repos *under this org*), so syncing one tenant cannot delete
  another's data; optional cascade deletes children of a deleted parent.
- **Takeaway:** full re-read each run; no conditional requests; correctness of
  deletion depends on having fetched the complete scope.

### Incremental ELT-style syncers (cursor + dedupe)

- **Model:** one table/stream per object type (issues, comments, review
  comments, commits, …), primary key = GitHub `id`.
- **Sync:** per-stream cursor on `updated_at`; each run requests with `since` =
  last cursor and stores the max `updated_at` seen. Boundary records are
  re-emitted ("each incremental sync re-emits the single record whose cursor
  value is exactly the timestamp the previous sync stopped at"), so
  append-plus-dedupe on the primary key is required.
- **Staleness:** updates are picked up; **deletions are not detected** by the
  cursor (a deleted comment simply stops appearing). Rate limits handled by
  rotating tokens and sleeping until reset.

### Knowledge-graph style GitHub importers (issue/PR content)

The common shape when the goal is "talk to the repo": `Repository`, `Issue`,
`PullRequest`, `Comment`, `User`, `Label` (sometimes `Commit`, `Milestone`)
with `(:User)-[:AUTHORED|OPENED]->(:Issue)`, `(:Issue)-[:IN_REPO]->(:Repository)`,
`(:Comment)-[:ON]->(:Issue)`, `(:Issue)-[:HAS_LABEL]->(:Label)`, plus
cross-references parsed from text (`#123`, "closes #123"). Issues and PRs are
merged on `(repo, number)` or on `node_id`; bodies/comments then feed chunking
and embedding. Freshness is usually a periodic re-run keyed on `updated_at`
(lower confidence: this section is a synthesis of the pattern, not a single
documented system).

## 5. Implications for the Resource identity key and staleness check

- **Identity key:** use the GraphQL `node_id` as the Resource's identity when
  it is available (any API response carries it), and keep `owner/repo#n` + URL
  as **alias/address properties** used for lookup. Do not key on
  `owner/repo#n`: renames, transfers and issue transfers change it, and a
  reused name silently points the old address at a different repo.
- **But the harness often only has an address.** A pasted URL or a `gh issue
  view` call output may not include `node_id`. The model needs an
  address → identity resolution step (MERGE on address, upgrade to `node_id`
  when first seen), and must handle two addresses converging on one identity
  after a rename/transfer.
- **Kind is not in the number:** an `owner/repo#n` must be resolved before you
  know Issue vs PullRequest; `node_id`'s type (or `pull_request` key) decides.
- **Staleness check:** store the source's `updated_at` (not `Last-Modified`)
  and our own `fetched_at` on the Resource. For a thread, issue `updated_at`
  covers its comments. If the agent is authenticated, also store the ETag for
  free revalidation; if not, revalidation costs a request, so a TTL on
  `fetched_at` (serve from cache within TTL) is the only zero-cost option.
- **Bulk sweeps:** model a sweep (e.g. "500 issues of repo X") as a scoped
  re-read; `since=<max cached updated_at>` gives the delta cheaply. Deletions
  will not show up in a `since` delta — if removal matters, borrow the
  mark-and-sweep pattern scoped to the repo, run only on complete sweeps.
- **Shared, not owned:** public Resources are identical for every user, so the
  `node_id` key lets two users' sessions converge on one Resource node — but
  authenticated vs unauthenticated reads can differ in content (and ETag), so
  record which mode fetched it if that ever matters.

## Open questions

1. Does an issue's `node_id` survive an issue transfer, or does GitHub mint a
   new issue object? Docs are silent; needs a probe on a transferred issue.
2. Do reactions, label changes, and PR review comments bump the issue/PR
   `updated_at`? (Comments do.) Determines whether one timestamp covers the
   whole cached thread for PRs.
3. When the harness fetches unauthenticated (60/h), should the Resources
   component ever revalidate on its own, or only serve cache + mark stale and
   let the agent decide to re-fetch?
4. How to resolve address → `node_id` for pasted text with no API response at
   hand — defer until the next fetch, or accept an address-keyed Resource that
   later merges?
