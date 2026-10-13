"""#534: does a Textual app start behind `cat shim | bash`, with and without reopening /dev/tty, and with no terminal?

Run with a Python that has textual installed. Needs `script` (macOS/Linux). Expected output is in the note.
"""

import os
import pathlib
import subprocess
import sys
import tempfile

APP = pathlib.Path(__file__).with_name("_app.py")
SHIMS = {
    "pty, shim reopens /dev/tty": (
        f'if (exec </dev/tty) 2>/dev/null; then exec "{sys.executable}" "{APP}" "$OUT" </dev/tty; '
        'else echo "NO TTY: plain mode" >>"$OUT"; fi\n',
        True,
    ),
    "pty, stdin left as the pipe": (f'exec "{sys.executable}" "{APP}" "$OUT"\n', True),
    "no terminal, shim probes /dev/tty": (
        f'if (exec </dev/tty) 2>/dev/null; then exec "{sys.executable}" "{APP}" "$OUT" </dev/tty; '
        'else echo "NO TTY: plain mode" >>"$OUT"; fi\n',
        False,
    ),
}

with tempfile.TemporaryDirectory() as tmp:
    for i, (name, (shim, with_pty)) in enumerate(SHIMS.items()):
        shim_path = pathlib.Path(tmp, f"shim{i}.sh")
        shim_path.write_text(shim)
        out = pathlib.Path(tmp, f"out{i}.txt")
        command = f"cat {shim_path} | bash"
        argv = ["script", "-q", "/dev/null", "bash", "-c", command] if with_pty else ["bash", "-c", command]
        proc = subprocess.run(
            argv,
            env={**os.environ, "OUT": str(out)},
            capture_output=True,
            timeout=30,
            start_new_session=True,  # no controlling terminal unless `script` makes one
            stdin=subprocess.DEVNULL,
        )
        print(f"--- {name}: exit {proc.returncode}\n{out.read_text().strip() if out.exists() else '(app never wrote)'}")
