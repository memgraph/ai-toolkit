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
