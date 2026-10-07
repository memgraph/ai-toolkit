# PROTOTYPE — GitHub Resource graph model (throwaway)

Question (map #454, ticket #460): what graph shape holds public GitHub Resources, Listings and
private Touches so that re-reading the same issues is served from memory?

```
docker run -d --name ai-toolkit-460-proto-wipe-me-7771 -p 7771:7687 memgraph/memgraph:latest
uv run --no-project --with neo4j python context-graph/resources-graph/prototype_graph_model/demo.py 1000   # scripted run
uv run --no-project --with neo4j python context-graph/resources-graph/prototype_graph_model/tui.py         # drive by hand
```

Needs `gh` logged in (stands in for the Sweep's optional config-file token). Wipes the scratch DB on each demo run.

- `address.py` — pure: parse URL / `owner/repo#n` / `gh` argv into an Address; listing subsumption.
- `github.py` — Sweep fetches (GraphQL, public-only) and the `updatedAt` probe.
- `store.py` — the graph model: hook-side Touch, `resource(address)` cache read, Sweep upserts.
- `sweep.py`, `demo.py`, `tui.py` — drivers.

Verdict and measurements are on the ticket's resolution comment.
