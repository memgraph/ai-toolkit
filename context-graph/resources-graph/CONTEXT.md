# Resources Graph

Remembers external resources an agent explored — GitHub first, later webpages, PDFs, pasted text — as shared canonical content, so re-reading them can be served from memory. Designed, not implemented: map [#454](https://github.com/memgraph/ai-toolkit/issues/454).

## Language

**Resource**:
Canonical, publicly readable external content with a stable identity (for GitHub, the GraphQL `node_id`) — e.g. a Repository, Issue or PullRequest. Belongs to no user.
_Avoid_: Source, document, reference, page (each names a narrower or already-taken concept)

**Resource Kind**:
The platform-specific type of a Resource — Repository, Issue, PullRequest for GitHub; later Page, PDF. Resource is the platform-neutral concept.
_Avoid_: Resource type (clashes with Entity Type)

**Address**:
How a resource was referred to when touched — a URL, `owner/repo#n`, `gh` arguments, listing filters. Mutable (renames, transfers); never a Resource's identity.
_Avoid_: Resource id, key

**Touch**:
One act of an agent exploring a resource: an Address, its provenance and time, private to the user whose session it belongs to. A listing is one Touch on its Repository, not one per member.
_Avoid_: Read, access, visit, fetch (a Touch carries no content)

**Unresolved Touch**:
A Touch whose Address the Sweep could not turn into a Resource, kept with a reason: not found, not public, or rate-limited.
_Avoid_: Placeholder resource, failed resource

**Sweep**:
Out-of-band process that resolves Touches' Addresses into Resources, fetches their canonical content, and links Touches to the Actions and Agents they came from. Never runs inside a hook.
_Avoid_: Sync, crawl, ingestion, reconciliation (Session Reconciliation is a different process)

**Provenance**:
How an Address reached the agent: **FETCHED** — the agent's own tool call, evidence of a model decision to explore; **PROMPTED** — it appeared in prompt text, with no claim that a human sent it.
_Avoid_: Handed by user, user-provided (prompts also come from scheduled tasks, subagents, other sessions)

## Relationships

- A **Touch** belongs to a Session; it is private to that Session's user via `(:User)-[:HAD_SESSION]->(:Session)`.
- A resolved **Touch** points at one **Resource**; many Touches, across users, share one Resource.
- A **Touch** links to the Action that caused it, and to the Agent when it happened inside a subagent.
- A listing **Touch** points at a Repository; the **Sweep** expands the listing into member Resources without creating Touches for them.
- An **MCP resource** is one channel through which a Resource can be touched, not a separate concept.
- unstructured2graph's **Source** is an ingestion's input Address; a Resource is the persisted, identified thing behind it.

## Flagged ambiguities

- "Resource" vs **Memory**: a Resource is canonical external content nobody asserted; a Memory is a user-owned assertion. "Every Memory is user-owned" does not extend to Resources.
- "What the agent saw" (a WebFetch summary, truncated shell output) is not Resource content; it stays on its ToolResult Action.
