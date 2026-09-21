"""Aggregate reads over the #350 prototype output.

Answers, per arm: does the answer-bearing edge survive; how much of the edge
set is label noise (one pair, several relation labels); what actually lands in
the `User` type, which is what #347's "collapse every User mention onto the
session's user node" rule would act on.
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
rows = json.loads((HERE / "constrained_edges_output.json").read_text())

FIRST_PERSON = {"i", "me", "my", "myself", "mine", "user", "i'm", "i've"}
ARMS = ("constrained", "permissive")


def norm(text: str) -> str:
    return " ".join(text.strip().lower().split())


print("=" * 100)
print("1. ANSWER SURVIVAL -- does an edge exist that carries the expected answer?")
print("=" * 100)
for row in rows:
    answer_tokens = {t for t in re.findall(r"[a-z0-9']+", str(row["answer"]).lower()) if len(t) > 3}
    print(f"\n[{row['question_type']}] {row['question']}")
    print(f"  answer: {row['answer']!r}  (session {row['session_id']})")
    for arm in ARMS:
        hits = []
        for edge in row["arms"][arm]["edges"]:
            text = f"{norm(edge['head_text'])} {norm(edge['tail_text'])}"
            if answer_tokens and any(token in text for token in answer_tokens):
                hits.append(
                    f"{edge['relation']}({edge['head_type']}:{edge['head_text']!r} -> "
                    f"{edge['tail_type']}:{edge['tail_text']!r})"
                )
        unique = sorted(set(hits))
        print(f"  {arm:<12} {len(unique)} answer-bearing edge(s)")
        for hit in unique[:6]:
            print(f"      {hit}")

print()
print("=" * 100)
print("2. LABEL NOISE -- one (head, tail) pair carrying several relation labels")
print("=" * 100)
for arm in ARMS:
    pairs: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        for edge in row["arms"][arm]["edges"]:
            pairs[(norm(edge["head_text"]), norm(edge["tail_text"]))].add(edge["relation"])
    multi = {k: v for k, v in pairs.items() if len(v) > 1}
    print(
        f"\n{arm}: {len(pairs)} distinct pairs, {len(multi)} carry >1 relation label "
        f"({100 * len(multi) / max(len(pairs), 1):.0f}%)"
    )
    for (head, tail), labels in sorted(multi.items(), key=lambda kv: -len(kv[1]))[:10]:
        print(f"      {head!r} -> {tail!r}: {sorted(labels)}")

print()
print("=" * 100)
print("3. WHAT LANDS IN `User` -- the type #347 would collapse onto the session's user node")
print("=" * 100)
for arm in ARMS:
    surface: Counter[str] = Counter()
    # identity_shape() keeps only aggregates, so count User surface forms from
    # the edges' own endpoints instead.
    for row in rows:
        for edge in row["arms"][arm]["edges"]:
            if edge["head_type"] == "User":
                surface[norm(edge["head_text"])] += 1
            if edge["tail_type"] == "User":
                surface[norm(edge["tail_text"])] += 1
    third_party = {k: v for k, v in surface.items() if k not in FIRST_PERSON and k != "assistant"}
    assistant = surface.get("assistant", 0)
    print(f"\n{arm}: {sum(surface.values())} User endpoints over {len(surface)} distinct surface forms")
    print(f"      first-person forms: {sum(v for k, v in surface.items() if k in FIRST_PERSON)}")
    print(f"      literal 'assistant' (the role prefix of the other speaker): {assistant}")
    print(f"      neither -- third parties typed as User: {sum(third_party.values())} over {len(third_party)} forms")
    for form, count in sorted(third_party.items(), key=lambda kv: -kv[1])[:12]:
        print(f"          {form!r} x{count}")

print()
print("=" * 100)
print("4. RELATION LABEL DISTRIBUTION")
print("=" * 100)
for arm in ARMS:
    labels: Counter[str] = Counter()
    for row in rows:
        for edge in row["arms"][arm]["edges"]:
            labels[edge["relation"]] += 1
    print(f"\n{arm}: " + ", ".join(f"{k}={v}" for k, v in labels.most_common()))

print()
print("=" * 100)
print("5. SELF-LOOPS AND DEGENERATE EDGES")
print("=" * 100)
for arm in ARMS:
    self_loops = []
    for row in rows:
        for edge in row["arms"][arm]["edges"]:
            if norm(edge["head_text"]) == norm(edge["tail_text"]):
                self_loops.append(
                    f"{edge['relation']}({edge['head_type']}:{edge['head_text']!r} -> "
                    f"{edge['tail_type']}:{edge['tail_text']!r})"
                )
    print(f"\n{arm}: {len(self_loops)} edges whose endpoints have identical text")
    for item in sorted(set(self_loops))[:10]:
        print(f"      {item}")

print()
print("=" * 100)
print("6. FRAGMENTATION (#346) -- span-scoped vs merged identity, summed over sessions")
print("=" * 100)
for arm in ARMS:
    mentions = sum(r["arms"][arm]["identity"]["mentions_total"] for r in rows)
    span_scoped = sum(r["arms"][arm]["identity"]["nodes_span_scoped"] for r in rows)
    merged = sum(r["arms"][arm]["identity"]["nodes_merged"] for r in rows)
    merged_optin = sum(r["arms"][arm]["identity"]["nodes_merged_mergeable_types_only"] for r in rows)
    print(
        f"\n{arm}: {mentions} mentions -> {span_scoped} span-scoped nodes -> {merged} merged nodes "
        f"({merged_optin} of them in opted-in types)"
    )
    print(f"      collapse factor: {mentions / max(merged, 1):.1f} mentions per merged node")
    per_type: Counter[str] = Counter()
    for row in rows:
        per_type.update(row["arms"][arm]["identity"]["mentions_per_type"])
    print("      mentions per type: " + ", ".join(f"{k}={v}" for k, v in per_type.most_common()))
    degree: Counter[str] = Counter()
    for row in rows:
        for item in row["arms"][arm]["identity"]["top_degree_merged"]:
            degree[f"{item['type']}:{item['text']}"] += item["degree"]
    print("      highest-degree merged nodes: " + ", ".join(f"{k}({v})" for k, v in degree.most_common(8)))
