"""Tests for reading and writing the committed corpus file.

deepeval's own ``EvaluationDataset.save_as("jsonl")`` is deliberately not used:
it writes only input/actual_output/expected_output/retrieval_context/context,
dropping ``additional_metadata`` (which carries the tier), ``name`` and
``source_file`` (upstream traceability), and joining ``context`` into a
"|"-delimited string.
"""

from typing import Any

from context_graph_eval.corpus import read_corpus, write_corpus
from deepeval.dataset import Golden


def _golden(**overrides):
    # Annotated because this is a kwargs bag: without it the literal infers
    # dict[str, str | int] and every str-typed parameter rejects the int.
    fields: dict[str, Any] = {
        "input": "What breed is the dog I adopted?",
        "expected_output": "A beagle.",
        "context": ["user: I adopted a beagle named Max"],
        "name": "gpt4_1a2b3c",
        "source_file": "longmemeval-v1",
        "additional_metadata": {"tier": 1, "question_type": "single-session-user"},
    }
    fields.update(overrides)
    return Golden(**fields)


def test_a_written_corpus_round_trips_every_field(tmp_path):
    path = tmp_path / "tier1.jsonl"
    write_corpus([_golden()], path)

    (restored,) = read_corpus(path)

    assert restored.input == "What breed is the dog I adopted?"
    assert restored.expected_output == "A beagle."
    assert restored.context == ["user: I adopted a beagle named Max"]
    assert restored.name == "gpt4_1a2b3c"
    assert restored.source_file == "longmemeval-v1"
    assert restored.additional_metadata == {"tier": 1, "question_type": "single-session-user"}


def test_context_entries_survive_a_pipe_character(tmp_path):
    """deepeval's own writer joins context on "|", which would split this entry
    into two. Ours must not."""
    path = tmp_path / "tier1.jsonl"
    write_corpus([_golden(context=["user: I run `ps aux | grep memgraph` daily"])], path)

    (restored,) = read_corpus(path)

    assert restored.context == ["user: I run `ps aux | grep memgraph` daily"]


def test_one_line_per_golden(tmp_path):
    path = tmp_path / "tier1.jsonl"
    write_corpus([_golden(name="a"), _golden(name="b"), _golden(name="c")], path)

    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 3


def test_write_reports_how_many_goldens_it_wrote(tmp_path):
    written = write_corpus([_golden(name="a"), _golden(name="b")], tmp_path / "tier1.jsonl")

    assert written == 2


def test_corpus_is_written_in_a_stable_order(tmp_path):
    """The corpus is committed to git, so byte-identical inputs must produce a
    byte-identical file -- otherwise every regeneration shows a spurious diff."""
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    goldens = [_golden(name="a"), _golden(name="b")]

    write_corpus(goldens, first)
    write_corpus(goldens, second)

    assert first.read_bytes() == second.read_bytes()
