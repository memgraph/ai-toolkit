"""Exchange dependency initialization at the real offline tokenizer boundary."""

import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("warm", [False, True])
def test_offline_encoding_initializes_before_retrieval(tmp_path, warm):
    env = {**os.environ, "TIKTOKEN_CACHE_DIR": str(tmp_path)}
    if warm:
        subprocess.run(
            [sys.executable, "-c", "import tiktoken; tiktoken.get_encoding('cl100k_base')"],
            env=env,
            check=True,
            timeout=60,
        )
    script = """
import socket

def offline(*args, **kwargs):
    raise OSError("test network disabled")

socket.socket.connect = offline
from sessions_graph import RecallConfig
from sessions_graph.recall import recall
from sessions_graph.recall_exchanges import prepare
"""
    if warm:
        script += "\nassert prepare(RecallConfig()).turns_k >= 24\n"
    else:
        # None cannot embed/query: this also checks failure precedes all graph work.
        script += """
try:
    recall(None, "user", "question", config=RecallConfig())
except RuntimeError as exc:
    assert "TIKTOKEN_CACHE_DIR" in str(exc), str(exc)
else:
    raise AssertionError("Empty offline cache unexpectedly succeeded")
"""
    subprocess.run([sys.executable, "-c", script], env=env, check=True, timeout=30)
