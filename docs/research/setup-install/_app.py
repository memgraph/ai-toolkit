"""The Textual app the launch probes run: logs that it mounted, then exits on a timer or on `q`."""

import sys
from typing import ClassVar

from textual.app import App
from textual.widgets import Static

OUT = sys.argv[1]
WAIT_FOR_KEY = len(sys.argv) > 2 and sys.argv[2] == "keys"


def log(line: str) -> None:
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(line + "\n")


class Probe(App[None]):
    BINDINGS: ClassVar = [("q", "done", "done")]

    def compose(self):
        yield Static("probe")

    def on_mount(self) -> None:
        log(f"MOUNTED stdin_tty={sys.stdin.isatty()} stdout_tty={sys.stdout.isatty()} size={self.size}")
        if not WAIT_FOR_KEY:
            self.set_timer(0.3, self.exit)

    def action_done(self) -> None:
        log("GOT KEY q")
        self.exit()


Probe().run()
