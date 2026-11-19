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

## For admins: a server

`compose` prints a docker-compose file for a spaces server. Its `memgraph` service
runs a licensed Memgraph with MAGE, which recall's embeddings need, and its memory
limit stays below the container's. Its `worker` service is the shared-space worker
(see below):

```bash
context-graph-spaces compose --bind 10.0.0.5 --port 7687 --memory-gib 4 > docker-compose.yml
cat > .env <<'ENV'
MEMGRAPH_ENTERPRISE_LICENSE=...
MEMGRAPH_ORGANIZATION_NAME=...
ENV
docker compose up -d
```

`--bind` is the host address Bolt listens on. Use a private network's address,
not `0.0.0.0`.

### The admin login

Provisioning needs a login with Memgraph's built-in `admin` role, or the
privileges `init` and `apply` use granted directly. Many admins created before
multi-tenancy are set up the second way. A directly granted login also needs
access to every database (`GRANT DATABASE *`), because `apply` writes into every
personal database it creates.

`init` creates that login on a fresh instance, then checks the instance can host
spaces:

```bash
context-graph-spaces --url bolt://memgraph.corp:7687 init      # --admin cgadmin is the default
```

It checks four things:

- the Enterprise licence
- the admin's privileges
- that MAGE's `embeddings.text` loads the model recall embeds with (it says why
  when it can't)
- Memgraph's memory limit

The new admin's password is printed once, as an `export` line for the commands
below.

On an instance that already has users, `init` creates nothing. Pass it the
existing admin (`--user`/`--password` or `MEMGRAPH_USER`/`MEMGRAPH_PASSWORD`) to
run the checks. An existing admin gives another login the role, or the
privileges directly:

```cypher
SET ROLE FOR cgadmin TO admin;
-- or
GRANT CONFIG, STATS, AUTH, MULTI_DATABASE_EDIT, MULTI_DATABASE_USE, PROFILE_RESTRICTION,
      MATCH, CREATE, MERGE, SET, DELETE, REMOVE, INDEX, CONSTRAINT TO cgadmin;
GRANT DATABASE * TO cgadmin;
```

The admin can't be named `admin`, which is the built-in role's name, or start with
`u_`, `space_` or `cg_`, which `apply` owns. If the login falls short, `init`,
`plan` and `apply` name the missing privilege and print the lines above, not a
stack trace.

### On Kubernetes

The MAGE image ships no embedding model. The first `embeddings.text` call
downloads the one recall uses into `$HOME/.cache/huggingface`. The
`memgraph-high-availability` Helm chart runs data instances with
`readOnlyRootFilesystem: true`, so that download fails and recall times out in
every space. Point the cache at the data volume on every data instance:

```yaml
extraEnv:
  data:
    - name: HF_HOME
      value: /var/lib/memgraph/huggingface   # persistent: survives restarts and failovers
```

If `extraEnv.data` is already set, add this entry to it. Each data pod needs
outbound HTTPS to huggingface.co for its first download, about 130 MB per pod.
Without that access, pre-seed the directory and set `HF_HUB_OFFLINE=1`. Every data
instance needs the model, not just the current main, or recall breaks at the next
failover. `init` checks only the instance it reaches, which on a cluster is the
main.

### The spec

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
export MEMGRAPH_USER="cgadmin"
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

Each setup line moves all of that member's agents to the team Memgraph. A member
who wants only some of them there adds `--agent <runtime>` (repeatable). The login
then goes into a [profile](../memgraph-context-graph/docs/configuration.md#profiles-one-agents-memory-somewhere-else),
and the other agents keep their Memgraph.

The setup lines are the only copy of the new passwords. If `apply` fails partway,
it still prints the lines for logins it already created, then the error. Run it
again once the error is fixed, and it continues from where it stopped.

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

On a server from `compose`, add `CG_SERVICE_PASSWORD=...` to `.env`, put the spec
where `--spec` said (`./spaces.toml` by default), and start it:

```bash
docker compose --profile worker up -d
```

It installs a CPU-only torch and keeps its packages and model on a volume, so
only the first start downloads them.

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
