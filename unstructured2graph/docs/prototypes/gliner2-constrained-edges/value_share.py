"""#361 item 1: what share of the corpus's answers are values rather than links?

Hand classification over all 100 committed goldens (indices below). Regex
cannot do this -- "four", "3.5 weeks" and "8 am - 4 pm (Day Shift)" defeat it,
and the judgement is the point -- so every answer is printed with its class
for audit. No model run.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

CORPUS = Path(__file__).resolve().parents[4] / "context-graph/eval/corpus/tier1-longmemeval.jsonl"

# Values, split by what answering them takes.
STATED = {1, 21, 23, 29, 30, 39, 42, 45, 46, 52, 54, 56, 60, 70, 71, 76, 77, 87}  # a literal the text contains
DATE_ARITHMETIC = {2, 5, 19, 20, 28, 36, 37, 47, 48, 53, 63, 69, 78, 80, 82, 84, 85, 97}
SUM_OF_VALUES = {11, 22, 27, 31, 50, 64, 67, 86}
COUNT_OF_EDGES = {8, 9, 15, 25, 40, 41, 43, 83, 88, 89, 91, 94}  # needs no value slot at all
NAMED = {4, 7, 16, 18, 26, 32, 35, 51, 55, 65, 66, 68, 72, 73, 74, 75, 81, 92, 96, 98, 99}
NARRATIVE = {6, 10, 12, 14, 17, 24, 44, 57, 58, 59, 62, 79, 90, 93, 95}

CLASSES = (
    ("stated literal", STATED),
    ("date arithmetic", DATE_ARITHMETIC),
    ("sum of stated values", SUM_OF_VALUES),
    ("count of edges", COUNT_OF_EDGES),
    ("named entity", NAMED),
    ("narrative", NARRATIVE),
)
VALUE_CLASSES = {"stated literal", "date arithmetic", "sum of stated values", "count of edges"}


def kind(index, abstention):
    if abstention:
        return "abstention"
    hits = [name for name, members in CLASSES if index in members]
    assert len(hits) == 1, (index, hits)
    return hits[0]


def main():
    rows = [json.loads(line) for line in CORPUS.open()]
    assert len(rows) == 100
    counts, by_type = Counter(), defaultdict(Counter)
    for i, row in enumerate(rows):
        k = kind(i, row["additional_metadata"]["abstention"])
        counts[k] += 1
        by_type[row["additional_metadata"]["question_type"]][k] += 1
    answerable = 100 - counts["abstention"]

    print(f"{answerable} answerable of 100 ({counts['abstention']} abstention)\n")
    for name, _ in CLASSES:
        print(f"  {name:<22}{counts[name]:>4}  {100 * counts[name] / answerable:>4.0f}% of answerable")
    value = sum(counts[c] for c in VALUE_CLASSES)
    slot = value - counts["count of edges"]
    print(f"\n  a value of some kind   {value:>4}  {100 * value / answerable:>4.0f}%")
    print(f"  needs a value slot     {slot:>4}  {100 * slot / answerable:>4.0f}%   (stated + date arithmetic + sums)")
    print(
        f"  relations + identity   {counts['count of edges']:>4}  {100 * counts['count of edges'] / answerable:>4.0f}%   (count of edges)"
    )

    ago = sorted(i for i in DATE_ARITHMETIC if "ago" in rows[i]["input"].lower())
    print(
        f"\n  date arithmetic: {len(ago)} 'how long ago' (valid_at suffices), "
        f"{len(DATE_ARITHMETIC) - len(ago)} 'between X and Y' (need the discussed event's date)"
    )

    order = [n for n, _ in CLASSES] + ["abstention"]
    print(f"\n  {'type':<27}" + "".join(f"{n.split()[0][:8]:>9}" for n in order))
    for qtype, c in sorted(by_type.items(), key=lambda kv: -sum(kv[1].values())):
        print(f"  {qtype:<27}" + "".join(f"{c[n]:>9}" for n in order))

    print("\nevery answer, with its class:")
    for i, row in enumerate(rows):
        m = row["additional_metadata"]
        print(
            f"  {i:>3} {kind(i, m['abstention']):<21} {m['question_type']:<26} "
            f"Q: {' '.join(row['input'].split())[:60]}  A: {' '.join(row['expected_output'].split())[:50]}"
        )


if __name__ == "__main__":
    main()
