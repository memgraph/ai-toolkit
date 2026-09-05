"""Read and write the committed corpus file.

The corpus is deepeval ``Golden`` records as JSONL, committed to git -- not held
in Memgraph. The corpus is the answer key, and the schema-evolution loop's whole
job is mutating the graph; an answer key living in that graph could be silently
invalidated by a migration, leaving no way to tell a real score regression from
a corrupted fixture. Git also proves the corpus did not change between the two
runs a cross-version comparison is comparing.

deepeval's own ``EvaluationDataset.save_as("jsonl")`` is not used. It writes
only input/actual_output/expected_output/retrieval_context/context, so it drops
``additional_metadata`` (carrying the tier, without which Tier 1 and Tier 2
cannot be scored separately), drops ``name``/``source_file`` (upstream
traceability), and joins ``context`` with "|" -- which corrupts any entry
containing that character.
"""

import json
from collections.abc import Iterable
from pathlib import Path

from deepeval.dataset import Golden

#: Written fields, in this order. Explicit rather than derived from Golden's own
#: field list so an upstream deepeval change cannot silently reshape a committed
#: corpus.
_FIELDS = (
    "input",
    "expected_output",
    "context",
    "name",
    "source_file",
    "additional_metadata",
)


def write_corpus(goldens: Iterable[Golden], path: Path) -> int:
    """Write goldens to ``path`` as JSONL. Returns how many were written.

    Output is deterministic: identical input produces a byte-identical file, so
    regenerating a committed corpus shows no spurious diff.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for golden in goldens:
            record = {field: getattr(golden, field) for field in _FIELDS}
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def read_corpus(path: Path) -> list[Golden]:
    """Read a corpus file written by :func:`write_corpus`."""
    with Path(path).open(encoding="utf-8") as handle:
        return [Golden(**json.loads(line)) for line in handle if line.strip()]
