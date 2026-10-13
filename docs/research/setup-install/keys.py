"""#534: type `q` into a real pty while the app runs under `cat shim | bash`. Does the key reach Textual?

Run with a Python that has textual installed (macOS/Linux). Expected output is in the note.
"""

import os
import pathlib
import pty
import select
import sys
import tempfile
import time

APP = pathlib.Path(__file__).with_name("_app.py")
SHIMS = {
    "shim reopens /dev/tty": f'exec "{sys.executable}" "{APP}" "$OUT" keys </dev/tty\n',
    "stdin left as the pipe": f'exec "{sys.executable}" "{APP}" "$OUT" keys\n',
}

with tempfile.TemporaryDirectory() as tmp:
    for i, (name, shim) in enumerate(SHIMS.items()):
        shim_path = pathlib.Path(tmp, f"shim{i}.sh")
        shim_path.write_text(shim)
        out = pathlib.Path(tmp, f"out{i}.txt")
        pid, fd = pty.fork()
        if pid == 0:
            os.environ["OUT"] = str(out)
            os.execvp("bash", ["bash", "-c", f"cat {shim_path} | bash"])
        deadline = time.monotonic() + 8
        sent = False
        while time.monotonic() < deadline:
            if not sent and out.exists() and "MOUNTED" in out.read_text():
                time.sleep(0.5)
                os.write(fd, b"q")
                sent = True
            if select.select([fd], [], [], 0.1)[0]:
                try:
                    os.read(fd, 65536)  # drain the screen so the app never blocks on output
                except OSError:
                    break
            if os.waitpid(pid, os.WNOHANG)[0]:
                break
        else:
            os.kill(pid, 9)
            os.waitpid(pid, 0)
        text = out.read_text().strip() if out.exists() else "(never mounted)"
        print(f"--- {name}: {text}{'' if 'GOT KEY' in text else '  -> key never arrived, killed'}")
