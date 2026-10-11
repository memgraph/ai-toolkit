# context-graph-spaces

The Enterprise edition of [Context Graph](../README.md): memory for many users on
one shared, licensed Memgraph. Each user gets a **personal space**, and teams get
**shared spaces**. Memgraph Enterprise enforces the isolation through
multi-tenancy, role-based access control and label- and property-based access
control. Our code doesn't filter.

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
export MEMGRAPH_URL=bolt://memgraph.corp:7687 MEMGRAPH_USER=<admin> MEMGRAPH_PASSWORD=...
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

## Tests

The tests run end-to-end against a licensed Memgraph. The licence is read from
`MEMGRAPH_ENTERPRISE_LICENSE` and `MEMGRAPH_ORGANIZATION_NAME` (the repo's `.env`
locally, secrets in CI):

```bash
./scripts/dev-memgraph.sh test-enterprise
```
