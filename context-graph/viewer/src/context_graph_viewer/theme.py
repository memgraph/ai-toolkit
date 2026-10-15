"""The viewer's look: Memgraph's brand colours as one Textual theme and a handful of Rich text styles.

Neutral first, orange second: the terminal is the brand's dark surface (``#231F20``) with white text, grays for
borders and secondary text, and orange only where the eye should go (focus, the selected row, key hints, the
timeline bars). Yellow, red and green appear only for warning, error and success. Every colour a view uses comes
from here, so a view never names one.
"""

from __future__ import annotations

from textual.theme import Theme

BLACK = "#231F20"
ORANGE = "#FB6E00"
WHITE = "#FFFFFF"
GRAY_SECONDARY = "#646265"
GRAY_MUTED = "#857F87"
SUCCESS = "#19AF66"
ERROR = "#DC2223"
WARNING = "#FFC500"

#: Rich styles for text built in the views.
MUTED = GRAY_MUTED
STRONG = f"bold {WHITE}"
OK = SUCCESS
BAD = ERROR
BAD_STRONG = f"bold {ERROR}"
WARN = WARNING
WARN_STRONG = f"bold {WARNING}"
ACCENT = ORANGE
#: Resource touches: not an alert, so white rather than a status colour.
TOUCH = f"bold {WHITE}"

NAME = "memgraph"

MEMGRAPH = Theme(
    name=NAME,
    primary=ORANGE,
    secondary=GRAY_MUTED,
    accent=ORANGE,
    warning=WARNING,
    error=ERROR,
    success=SUCCESS,
    foreground=WHITE,
    background=BLACK,
    surface=BLACK,
    panel=BLACK,
    dark=True,
    variables={
        "border": GRAY_SECONDARY,
        "border-blurred": GRAY_SECONDARY,
        "footer-key-foreground": ORANGE,
        "footer-description-foreground": GRAY_MUTED,
        "input-cursor-background": ORANGE,
        "input-selection-background": f"{ORANGE} 35%",
        "block-cursor-background": GRAY_SECONDARY,
        "block-cursor-foreground": WHITE,
        "block-cursor-text-style": "bold",
        "block-cursor-blurred-background": GRAY_SECONDARY,
        "block-cursor-blurred-foreground": WHITE,
        "scrollbar": GRAY_SECONDARY,
        "scrollbar-hover": GRAY_MUTED,
        "scrollbar-active": ORANGE,
        "scrollbar-background": BLACK,
        "scrollbar-background-hover": BLACK,
        "scrollbar-background-active": BLACK,
    },
)
