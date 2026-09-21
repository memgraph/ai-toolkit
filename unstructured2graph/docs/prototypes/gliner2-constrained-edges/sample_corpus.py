"""Pull a small, readable sample of real sessions out of the LongMemEval file.

The cached corpus is 277MB of JSON array; json.load()ing it to read six
sessions is wasteful, so records are decoded incrementally with raw_decode and
the scan stops as soon as enough question types are covered.

Session text is built exactly as sessions-graph's reconciliation builds it --
"role: content" per turn, deduped, joined with a blank line (core.py's
_PreparedSession.combined_text) -- so the prototype reads the same string a
real reconciliation would hand the extractor.
"""

import hashlib
import json
import sys
from pathlib import Path

CORPUS = Path.home() / ".cache/context-graph-eval/longmemeval-s-98d7416c24c7.json"
OUT = Path(__file__).parent / "sample_sessions.json"

# One question per type, so the sample spans the shapes the corpus actually
# asks -- #347 measured 42% of questions turning on time/change, and a sample
# of only single-session-user questions would hide exactly that.
WANTED_TYPES = [
    "single-session-user",
    "temporal-reasoning",
    "knowledge-update",
    "multi-session",
    "single-session-preference",
    "single-session-assistant",
]


def combined_text(turns: list[dict]) -> str:
    """The one document a session becomes, per sessions-graph reconciliation."""
    unique: dict[str, str] = {}
    for turn in turns:
        text = f"{turn['role']}: {turn['content']}".strip()
        if not text:
            continue
        unique.setdefault(hashlib.sha256(text.encode()).hexdigest(), text)
    return "\n\n".join(unique.values())


def stream_records(path: Path):
    """Yield top-level records of a JSON array without loading the whole file."""
    decoder = json.JSONDecoder()
    buffer = ""
    with path.open(encoding="utf-8") as handle:
        buffer = handle.read(1)  # leading '['
        assert buffer.strip() == "[", buffer
        buffer = ""
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                return
            buffer += chunk
            while True:
                stripped = buffer.lstrip()
                if stripped[:1] in (",", ""):
                    buffer = stripped[1:]
                    continue
                if stripped[:1] == "]":
                    return
                try:
                    record, end = decoder.raw_decode(stripped)
                except ValueError:
                    buffer = stripped
                    break
                buffer = stripped[end:]
                yield record


def main() -> int:
    picked: dict[str, dict] = {}
    for record in stream_records(CORPUS):
        qtype = record.get("question_type")
        if qtype not in WANTED_TYPES or qtype in picked:
            continue
        evidence_ids = set(record["answer_session_ids"])
        sessions = []
        for session_id, date, turns in zip(
            record["haystack_session_ids"],
            record["haystack_dates"],
            record["haystack_sessions"],
            strict=True,
        ):
            if session_id not in evidence_ids:
                continue
            sessions.append(
                {
                    "session_id": session_id,
                    "date": date,
                    "text": combined_text(turns),
                    "turns": len(turns),
                }
            )
        if not sessions:
            continue
        picked[qtype] = {
            "question_id": record["question_id"],
            "question_type": qtype,
            "question": record["question"],
            "question_date": record["question_date"],
            "answer": record["answer"],
            "sessions": sessions,
        }
        print(
            f"{qtype:28} {record['question_id']}  "
            f"{len(sessions)} evidence session(s), "
            f"{sum(len(s['text']) for s in sessions)} chars",
            flush=True,
        )
        if len(picked) == len(WANTED_TYPES):
            break

    OUT.write_text(json.dumps([picked[t] for t in WANTED_TYPES if t in picked], indent=2), encoding="utf-8")
    print(f"\nwrote {OUT} ({OUT.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
