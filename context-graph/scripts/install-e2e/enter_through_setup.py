"""Run a command on a pseudo-terminal, pressing Enter at every question, as a new user taking each default would.

setup flushes anything typed before its first question is drawn, so Enter goes in only after "Your name"
shows and the output has gone quiet, and again each time it goes quiet, until setup starts setting up.
Every Enter takes the shown question's default, so an extra one can't change an answer.

Usage: python3 enter_through_setup.py --expect "text" ... -- bash ./context-graph/scripts/install.sh
Exits non-zero when the command fails or its output lacks an --expect text.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import pty
import re
import select
import signal
import struct
import sys
import termios
import time

QUIET_SECONDS = 1.0
# Wide enough that the plan's lines aren't cut to fit.
ROWS, COLUMNS = 50, 200
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--expect", action="append", default=[], help="text the output must contain (repeatable)")
    parser.add_argument("--timeout", type=float, default=900, help="seconds before the command is killed")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("no command given")

    pid, fd = pty.fork()
    if pid == 0:
        os.execvp(command[0], command)
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", ROWS, COLUMNS, 0, 0))

    output = b""
    quiet_since = time.monotonic()
    deadline = quiet_since + args.timeout
    while True:
        now = time.monotonic()
        if now > deadline:
            os.kill(pid, signal.SIGKILL)
            print(f"\nenter_through_setup: killed after {args.timeout:.0f}s", file=sys.stderr)
            return 1
        if select.select([fd], [], [], 0.2)[0]:
            try:
                chunk = os.read(fd, 4096)
            except OSError:  # EIO: the command exited and closed the terminal
                break
            if not chunk:
                break
            output += chunk
            sys.stdout.buffer.write(chunk)
            sys.stdout.flush()
            quiet_since = now
            continue
        text = output.decode(errors="replace")
        asking = "Your name" in text and "Setting up" not in text
        if asking and now - quiet_since > QUIET_SECONDS:
            os.write(fd, b"\r")
            quiet_since = now

    _, status = os.waitpid(pid, 0)
    code = os.waitstatus_to_exitcode(status)
    text = ANSI.sub("", output.decode(errors="replace"))
    missing = [expected for expected in args.expect if expected not in text]
    for expected in missing:
        print(f"enter_through_setup: output lacks {expected!r}", file=sys.stderr)
    return code or (1 if missing else 0)


if __name__ == "__main__":
    sys.exit(main())
