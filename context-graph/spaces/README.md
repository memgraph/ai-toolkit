# Context Graph Spaces

The Enterprise edition of [Context Graph](../README.md): memory for many users on
one shared, licensed Memgraph. Each user gets a **personal space**, and teams get
**shared spaces**. Memgraph Enterprise enforces the isolation through
multi-tenancy, role-based access control and label- and property-based access
control. Access is enforced by the database.

## Requirements and installation

- A licensed Memgraph Enterprise instance with multi-tenancy enabled.
- An admin login for provisioning users, databases, and access rules.
- A Context Graph installation for each user.

```bash
pip install context-graph-spaces
```

Install this package in each user's Context Graph tool environment as well as the environment running the admin commands.

## Configure a user

The Community edition (one user, one database) doesn't need this package and
doesn't change when it's installed. A user's config file selects Enterprise
explicitly:

```toml
edition = "enterprise"

[memgraph]
url = "bolt://memgraph.corp:7687"
user = "ante"        # your Memgraph login is your identity
password = "..."
```

## For admins: provisioning

Spaces are declared in a spec file and applied idempotently:

```toml
members = ["ante", "bob"]
personal_memory_limit_mb = 2048

[spaces.payments]
members = ["ante", "bob"]
memory_limit_mb = 4096
# labels = [...]       # what members may write; defaults to curated memory
# edge_types = [...]
```

```bash
export MEMGRAPH_URL="bolt://memgraph.corp:7687"
export MEMGRAPH_USER="admin"
export MEMGRAPH_PASSWORD="your-admin-password"
context-graph-spaces plan spaces.toml     # what would change
context-graph-spaces apply spaces.toml    # make it so; prints one-time setup lines for new logins
```

What `apply` maintains:

| Object | Name | Who |
|---|---|---|
| Personal database | `u_<login>` | its owner (main database), plus nobody else |
| Shared database | `space_<name>` | members: read everything, write only the spec's labels |
| Roles | `cg_u_<login>_<hash>`, `space_<name>_member_<hash>`, `cg_worker_<hash>` | named after what they grant |
| Service login | `cg_service` | writes Chunks and entities in shared spaces |
| Tenant profiles | `cg_<mb>mb` | memory caps from the spec |

Removing someone from the spec revokes their access. Removing a space revokes
access to it. Users and databases are only dropped with `--prune`. Only objects
with the `u_`, `space_` and `cg_` prefixes are ever touched.

## The shared-space worker

Members can write only curated memory into a shared space. The derived graph
there (chunks, entities, relations, and the embeddings recall uses) is written by
one service login, `cg_service`. Run the worker next to `apply`, with the
password `apply` printed once:

```bash
export CG_SERVICE_PASSWORD=...
context-graph-spaces reconcile spaces.toml --watch     # one pass every 30s
```

Each pass, for each shared space, it does two things:

1. It removes what deleted files left behind. A member's delete can't touch chunks.
2. It reconciles changed memory files with GLiNER2 under the space's ontology.
   No LLM is involved.

Personal spaces keep reconciling locally, as in Community.

## Tests

The tests run end-to-end against a licensed Memgraph. The licence is read from
`MEMGRAPH_ENTERPRISE_LICENSE` and `MEMGRAPH_ORGANIZATION_NAME` (the repo's `.env`
locally, secrets in CI):

```bash
./scripts/dev-memgraph.sh test-enterprise
```
