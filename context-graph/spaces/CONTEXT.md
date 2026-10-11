# Context Graph Spaces

The Enterprise edition of Context Graph: memory for many users on one shared,
licensed Memgraph, where Memgraph's own access control enforces the isolation
(map #501).

## Language

**Personal Space**:
A member's own database, `u_<login>`, and their main database. Hooks write all of the member's activity here. Only the member and nobody else can reach it.
_Avoid_: tenant, user database

**Shared Space**:
A database `space_<name>` that a set of members can read. Members may create and change only the labels and edge types the **Spaces Spec** allows; v1 allows curated memory only. A team or the whole org is just a Shared Space with different members.
_Avoid_: team space (a team is one use), workspace

**Spaces Spec**:
The admin's TOML file listing members and Shared Spaces. It is the single source of desired state. `context-graph-spaces plan` diffs Memgraph against it; `apply` makes Memgraph match.
_Avoid_: config (that's the member's own Context Graph config file)

**Space Membership**:
A `(:SpaceMembership {space, database})` node in a member's Personal Space, written by `apply`. It is how recall learns which Shared Spaces to read, because an unprivileged login can't list its own grants. Memgraph's grants remain the authority; a stale node only costs a permission error.

**Service Account**:
`cg_service`. Holds the worker role on every Shared Space and writes what members can't there, such as Chunks and entities from reconciliation.

**Owned Object**:
A database, role, user or tenant profile with a `u_`, `space_` or `cg_` prefix, or a user whose `u_<login>` database exists. The provisioner changes only owned objects.

## Relationships

- **Spaces Spec -> Memgraph**: `apply` creates databases, versioned roles (named by a hash of their grants), users, role assignments scoped to one database, tenant profiles and Space Memberships. Removing a member or a space revokes access; `--prune` also drops users and databases.
- **Context Graph config -> EnterpriseSpaces**: with `edition = "enterprise"`, agent-context-graph's `SpaceResolver` is this package's resolver, found through the `agent_context_graph.editions` entry point.

## Flagged ambiguities

- "Role": in this package, a Memgraph auth role (`space_payments_member_3fa2c1d0`). Not a message speaker role (`actions-graph`'s `MessageRole`).
