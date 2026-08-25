"""Build a committed eval corpus from a pinned upstream benchmark.

Fetches a pinned LongMemEval release, converts it to deepeval Goldens, and
writes the JSONL that gets committed. The downloaded file is a build artifact
and is not committed -- only the converted corpus is (see #302).
"""

import argparse
import tempfile
from pathlib import Path

from .convert.longmemeval import DEFAULT_REVISION, build_corpus, fetch, load_raw
from .corpus import write_corpus


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="context-graph-eval", description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    build = subcommands.add_parser(
        "build-corpus",
        help="fetch a pinned LongMemEval release and write the Tier 1 corpus",
    )
    build.add_argument(
        "--variant",
        default="s",
        help="LongMemEval variant: 's' (~40 sessions) or 'm' (~500). 'oracle' is refused: "
        "it has no distractor sessions, so retrieval would score well by construction.",
    )
    build.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help="pinned upstream revision. Changing this invalidates prior baselines.",
    )
    build.add_argument(
        "--limit",
        type=int,
        default=None,
        help="sample down to this many questions, spread across question types. "
        "Tier 1 starts small and scales only when the signal is too noisy to decide on.",
    )
    build.add_argument(
        "--out",
        type=Path,
        default=Path("corpus/tier1-longmemeval.jsonl"),
        help="where to write the corpus JSONL",
    )

    args = parser.parse_args(argv)

    if args.command == "build-corpus":
        return _build_corpus(args)
    return 1


def _build_corpus(args) -> int:
    with tempfile.TemporaryDirectory() as workdir:
        raw = fetch(args.variant, args.revision, dest=Path(workdir) / "upstream.json")
        records = load_raw(raw)
        print(f"fetched {len(records)} records from longmemeval-{args.variant} @ {args.revision[:12]}")

        goldens = build_corpus(records, limit=args.limit)
        written = write_corpus(goldens, args.out)
        print(f"wrote {written} goldens to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
