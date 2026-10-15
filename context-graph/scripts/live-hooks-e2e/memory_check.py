"""Seed and check the memory files a live memory run reads and writes.

Run through ``live-hooks-e2e.sh memory``, which points ``CONTEXT_GRAPH_CONFIG``
at the disposable instance's config, so the store is the one the agents' MCP
server writes to.

    memory_check.py seed                       write the file every agent reads back
    memory_check.py written <runtime> <word>   exit 1 unless the agent's file holds <word>
"""

import sys

from sessions_graph import SessionsGraph

from memgraph_context_graph.adapters._identity import load_config
from memgraph_context_graph.spaces import spaces_for

SEED_PATH = "/memories/notes/seed.md"
SEED = "---\ndescription: The e2e read codeword\ntype: reference\n---\nThe e2e read codeword is QUARTZ-PELICAN.\n"


def _store():
    config = load_config()
    # No setup(): the agents' MCP server already made the schema, and setup's index DDL would wait on the
    # hooks' detached embed jobs for exclusive access.
    graph = SessionsGraph(**spaces_for(config).personal().target.client_kwargs())
    if not config.user_id:
        raise SystemExit("the e2e config has no identity.user_id")
    return graph.memory_store(config.user_id)


def main(argv: list[str]) -> int:
    if argv[:1] == ["seed"]:
        _store().create(SEED_PATH, SEED)
        return 0
    if len(argv) == 3 and argv[0] == "written":
        _, runtime, word = argv
        path = f"/memories/notes/{runtime}.md"
        try:
            text = _store().view(path)
        except Exception as exc:
            print(f"{runtime}: no {path} ({exc})")
            return 1
        if word not in text:
            print(f"{runtime}: {path} doesn't hold {word}: {text!r}")
            return 1
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
