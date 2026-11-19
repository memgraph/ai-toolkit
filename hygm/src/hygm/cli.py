"""The `hygm` command line: validate, lint, convert, diff and resolve model files (docs/contract.md, 9.1).

A model file is read by its suffix: `.yaml`, `.yml` and `.json` as hygm model
files, anything else as OWL in a serialization rdflib reads. The exit code is
1 when a model fails validation or `resolve --check` finds its output stale;
lint findings never fail a run, and a usage problem (a missing file, an
unknown suffix to write) exits 2.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import yaml

from .errors import ModelError
from .evolution import diff, operation_to_mapping
from .lint import lint_model
from .resolved import resolved_to_json
from .strategies import DEFAULT_BASE, ManualStrategy, OwlImportStrategy, model_to_mapping, model_to_owl
from .types import HygmModel

_MODEL_FILES = (".yaml", ".yml", ".json")

#: The rdflib serialization each OWL suffix is written in.
_OWL_FORMATS = {".ttl": "turtle", ".owl": "xml", ".rdf": "xml", ".xml": "xml", ".jsonld": "json-ld", ".nt": "nt"}

_INVALID, _USAGE = 1, 2


class _UsageError(Exception):
    """A problem with the command, not with a model."""


def main(argv: Sequence[str] | None = None) -> int:
    """Run the `hygm` command line with `argv` (the process's arguments by default) and return its exit code."""
    args = _parser().parse_args(argv)
    try:
        return args.run(args)
    except ModelError as e:
        print(f"invalid: {e} [{', '.join(e.codes)}]", file=sys.stderr)
        return _INVALID
    except _UsageError as e:
        print(f"hygm: {e}", file=sys.stderr)
        return _USAGE


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hygm", description="Validate, lint, convert, diff and resolve hygm models.")
    commands = parser.add_subparsers(required=True, metavar="command")

    validate = commands.add_parser("validate", help="check a model file; exit 1 if it is invalid")
    validate.add_argument("file", type=Path)
    validate.set_defaults(run=_validate)

    lint = commands.add_parser("lint", help="report where a model departs from the design guidelines")
    lint.add_argument("file", type=Path)
    lint.add_argument("--json", action="store_true", help="print the findings as a JSON list")
    lint.set_defaults(run=_lint)

    convert = commands.add_parser("convert", help="write a model in another format, chosen by the output suffix")
    convert.add_argument("input", type=Path)
    convert.add_argument("output", type=Path)
    convert.add_argument("--base", default=DEFAULT_BASE, help="where OWL output mints IRIs (default: %(default)s)")
    convert.set_defaults(run=_convert)

    compare = commands.add_parser("diff", help="print the operations that turn one model into another, as JSON")
    compare.add_argument("a", type=Path)
    compare.add_argument("b", type=Path)
    compare.set_defaults(run=_diff)

    resolve = commands.add_parser("resolve", help="write a valid model in the resolved form, as JSON")
    resolve.add_argument("file", type=Path)
    resolve.add_argument("-o", "--output", type=Path, help="the file to write (default: standard output)")
    resolve.add_argument(
        "--check", action="store_true", help="write nothing; exit 1 if the output file isn't what resolving writes"
    )
    resolve.set_defaults(run=_resolve)
    return parser


def _load(path: Path) -> HygmModel:
    if not path.is_file():
        raise _UsageError(f"{path}: no such file")
    if path.suffix in _MODEL_FILES:
        return ManualStrategy().create_model(path)
    return OwlImportStrategy().create_model(path)


def _validate(args: argparse.Namespace) -> int:
    model = _load(args.file)
    print(f"valid: {len(model.node_types)} node types, {len(model.relation_types)} relation types")
    return 0


def _lint(args: argparse.Namespace) -> int:
    findings = lint_model(_load(args.file))
    if args.json:
        print(json.dumps([vars(f) for f in findings], indent=2, ensure_ascii=False))
        return 0
    for f in findings:
        print(f"{f.severity:7} {f.code:30} {f.element}: {f.message}; {f.suggestion}")
    print(f"{len(findings)} finding(s)")
    return 0


def _convert(args: argparse.Namespace) -> int:
    model, suffix = _load(args.input), args.output.suffix
    if suffix in (".yaml", ".yml"):
        text = yaml.safe_dump(model_to_mapping(model), sort_keys=False, allow_unicode=True)
    elif suffix == ".json":
        text = json.dumps(model_to_mapping(model), indent=2, ensure_ascii=False) + "\n"
    elif suffix in _OWL_FORMATS:
        text = model_to_owl(model, format=_OWL_FORMATS[suffix], base=args.base)
    else:
        known = ", ".join((*_MODEL_FILES, *_OWL_FORMATS))
        raise _UsageError(f"{args.output}: can't tell the format from {suffix!r}; use one of {known}")
    args.output.write_text(text, encoding="utf-8")
    return 0


def _diff(args: argparse.Namespace) -> int:
    operations = diff(_load(args.a), _load(args.b))
    print(json.dumps([operation_to_mapping(op) for op in operations], indent=2, ensure_ascii=False))
    return 0


def _resolve(args: argparse.Namespace) -> int:
    if args.check and args.output is None:
        raise _UsageError("--check needs the resolved file to compare against: -o <file>")
    text = resolved_to_json(_load(args.file))
    if args.check:
        written = args.output.read_text(encoding="utf-8") if args.output.is_file() else None
        if written != text:
            print(f"stale: {args.output} isn't {args.file} resolved; run hygm resolve without --check", file=sys.stderr)
            return _INVALID
        return 0
    if args.output is None:
        sys.stdout.write(text)
    else:
        args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
