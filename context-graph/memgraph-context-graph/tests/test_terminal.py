"""Which colours a terminal gets: what the environment says, except where a terminal is known to misread it."""

from __future__ import annotations

import pytest

from memgraph_context_graph.terminal import color_support


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({"COLORTERM": "truecolor"}, "truecolor"),
        ({"COLORTERM": "24bit"}, "truecolor"),
        ({"WT_SESSION": "1"}, "truecolor"),
        ({"TERM": "xterm-256color"}, "256"),
        ({"COLORTERM": "truecolor", "NO_COLOR": "1"}, ""),
        ({"TERM_PROGRAM": "iTerm.app", "COLORTERM": "truecolor"}, "truecolor"),
    ],
)
def test_the_environment_decides(env, expected):
    assert color_support(env, isatty=True, macos="14.6.1") == expected


def test_a_pipe_gets_no_colour():
    assert color_support({"COLORTERM": "truecolor"}, isatty=False) == ""


def test_apple_terminal_before_macos_26_is_256_colour_whatever_colorterm_says():
    env = {"TERM_PROGRAM": "Apple_Terminal", "COLORTERM": "truecolor"}

    assert color_support(env, isatty=True, macos="14.6.1") == "256"
    assert color_support(env, isatty=True, macos="15.2") == "256"
    assert color_support(env, isatty=True, macos="26.0") == "truecolor"
    assert color_support(env, isatty=True, macos="") == "256"
