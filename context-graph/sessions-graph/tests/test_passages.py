"""Splitting messages into passages, and choosing which to show."""

import pytest
from sessions_graph.passages import PASSAGE_CHARS, best_passage, excerpt, split_passages

LONG = (
    "We set up the project.\n\n"
    + "Background sentence about the stack. " * 60
    + "\n\nThe onboarding deadline moved to April 22. That is final."
)


def test_passages_are_consecutive_and_bounded():
    passages = split_passages(LONG)

    assert "".join(passages) == LONG
    assert len(passages) > 1
    assert all(len(passage) <= PASSAGE_CHARS for passage in passages)


def test_a_run_with_no_break_is_cut_at_the_bound():
    text = "x" * (PASSAGE_CHARS * 2 + 5)

    assert [len(passage) for passage in split_passages(text)] == [PASSAGE_CHARS, PASSAGE_CHARS, 5]


def test_a_short_message_is_one_passage():
    assert split_passages("Hi there.") == ["Hi there."]
    with pytest.raises(ValueError):
        split_passages("Hi", size=0)


def test_the_passage_holding_a_sentence_wins_over_word_overlap():
    passages = split_passages(LONG)

    assert best_passage(passages, "The onboarding deadline moved to April 22.") == len(passages) - 1
    assert best_passage(passages, "when did we set up the project") == 0


def test_a_short_turn_is_shown_whole_and_a_long_one_as_its_hits():
    assert excerpt("short text", [5], 100) == "short text"

    passages = split_passages(LONG)
    shown = excerpt(LONG, [len(passages) - 1], 1500)

    assert shown.startswith("… ")
    assert "April 22" in shown
    assert "We set up the project." not in shown
