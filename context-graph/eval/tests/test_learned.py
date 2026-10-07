"""The learned-ontology step: one `sessions-graph derive` per user, outcomes read from its output."""

import pytest
from context_graph_eval import learned

pytest.importorskip("sessions_graph")


class _Db:
    def __init__(self, users):
        self.users = users

    def query(self, cypher, params=None):
        return [{"user_id": user} for user in self.users]


@pytest.mark.asyncio
async def test_each_user_is_derived_once_and_its_outcome_read(monkeypatch, tmp_path):
    script = tmp_path / "sessions-graph"
    script.write_text(
        "#!/bin/sh\n"
        'case "$3" in\n'
        '  alice) echo "alice: adopted (milestone 2)";;\n'
        '  bob) echo "bob: rejected (milestone 2)";;\n'
        '  *) echo "boom" >&2; exit 1;;\n'
        "esac\n"
    )
    script.chmod(0o755)
    monkeypatch.setattr(learned.shutil, "which", lambda name: str(script))

    derived = await learned.derive_users(
        _Db(["alice", "bob", "carol"]), memgraph_url="bolt://x", seeds=1, workers=2, progress=False
    )

    assert derived.outcomes == {"alice": "adopted", "bob": "rejected"}
    assert derived.failures == {"carol": "boom"}
    assert derived.counts() == {"adopted": 1, "rejected": 1, "error": 1}
