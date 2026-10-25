"""Fusing several spaces' recall results into one: pure, no Memgraph."""

from sessions_graph.recall import MemoryNote, Recalled, Turn, fuse


def _notes(*texts, folder="/memories"):
    return [MemoryNote(path=f"{folder}/{text}.md", updated_at="2026-10-01", text=text) for text in texts]


def _turn(timestamp):
    return Turn(action_id=timestamp, session_id="s", timestamp=timestamp, speaker="user", text=timestamp)


def test_one_space_is_returned_untouched():
    alone = Recalled(question="q", memories=_notes("a", "b"))

    assert fuse("q", [alone], memories_k=1) is alone


def test_memories_interleave_by_rank_across_spaces():
    personal = Recalled(question="q", memories=_notes("p1", "p2", "p3"))
    shared = Recalled(question="q", memories=_notes("s1", "s2", folder="/memories/spaces/team"))

    fused = fuse("q", [personal, shared], memories_k=4)

    assert [note.text for note in fused.memories] == ["p1", "s1", "p2", "s2"]


def test_the_same_text_in_two_spaces_shows_once_as_the_shared_copy():
    personal = Recalled(question="q", memories=_notes("runbook", "other"))
    shared = Recalled(question="q", memories=_notes("runbook", folder="/memories/spaces/team"))

    fused = fuse("q", [personal, shared], memories_k=5)

    assert [note.path for note in fused.memories] == ["/memories/spaces/team/runbook.md", "/memories/other.md"]


def test_turns_stay_in_time_order_and_missing_spaces_are_said():
    personal = Recalled(question="q", turns=[_turn("2026-09-02"), _turn("2026-09-04")])
    shared = Recalled(question="q", turns=[_turn("2026-09-03")])

    fused = fuse("q", [personal, shared], memories_k=5, missing=("shared space sales (timeout)",))

    assert [turn.timestamp for turn in fused.turns] == ["2026-09-02", "2026-09-03", "2026-09-04"]
    assert "Not searched, so rows from them are missing: shared space sales (timeout)." in fused.render()
    assert fused.to_json()["missing_spaces"] == ["shared space sales (timeout)"]


def test_exchange_fusion_keeps_duplicate_ids_scoped_and_memory_budget_shared():
    personal_turns = [
        Turn("a", "personal", "2026-09-01", "user", "personal question"),
        Turn("b", "personal", "2026-09-02", "assistant", "personal answer"),
    ]
    shared_turns = [
        Turn("a", "shared", "2026-09-03", "user", "shared question"),
        Turn("b", "shared", "2026-09-04", "assistant", "shared answer"),
    ]
    personal = Recalled("q", turns=personal_turns, exchanges=(("a", "b"),), memories=_notes("p"))
    shared = Recalled("q", turns=shared_turns, exchanges=(("a", "b"),), memories=_notes("s"))

    fused = fuse("q", [personal, shared], memories_k=1)

    assert fused.space_results == (personal, shared)
    assert fused.lines() == [personal.memories[0].line(1500), *personal.lines()[1:], *shared.lines()[1:]]
    scoped = fused.to_json()["spaces"]
    assert [space["exchanges"] for space in scoped] == [[["a", "b"]], [["a", "b"]]]
    assert [space["turns"][0]["session_id"] for space in scoped] == ["personal", "shared"]
    assert len(fused.turns) == 4


def test_exchange_grouping_survives_missing_space_and_mixed_packing():
    grouped = Recalled("q", turns=[_turn("2026-09-02")], exchanges=(("2026-09-02",),))
    flat = Recalled("q", turns=[_turn("2026-09-01")])
    for results in ([grouped], [grouped, flat]):
        fused = fuse("q", results, memories_k=5, missing=("sales (timeout)",))
        assert "EXCHANGE" in fused.render()
        assert "sales (timeout)" in fused.render()
        assert all(turn.line(1500) in fused.lines() for result in results for turn in result.turns)
