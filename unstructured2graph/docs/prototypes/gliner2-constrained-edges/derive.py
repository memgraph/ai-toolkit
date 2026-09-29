"""#366: run #353's derivation contract once, end to end.

    propose  (LLM)     whole sessions, time-stratified, under a prompt budget
    observe  (GLiNER2) one permissive pass, one turn per window (#352)
    prune    (LLM)     narrow endpoints to observed pairs, set identity, drop
                       zero-instance types/relations; no renames, no additions
    validate           stand-in for hygm.validation's hard gate

`python derive.py A` and `python derive.py B` run two derivations from disjoint
propose samples (the stability check); both observe the same sample. Every
stage is cached in derivation/<name>.json, so a rerun never re-spends the LLM.
The LLM is `claude -p` under the caller's login, run with no user settings (so
no context-graph hooks record these calls) and no tools; its model, tokens and
cost are recorded as provenance.

The 10 evidence sessions (sample_sessions.json) are excluded from every sample:
read_derived.py evaluates on them.
"""

import json
import random
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import value_types as vt
from windowing import turn_windows, turns_of

HERE = Path(__file__).parent
OUT = HERE / "derivation"
SEED = 366
PROPOSE_STRATA, PROPOSE_CHARS = 10, 120_000
OBSERVE_SESSIONS = 60
# gliner2 2.0.0 cuts relation candidates to relation_pair_cap (128) and then
# max_edges_per_type (256), breaking score ties on str((entity_type, start, end)).
# A permissive endpoint gives every span one equal-scoring copy per type, so with
# more than 11 types the cut is alphabetical and User, sorting last, is always
# lost (0 User heads at the defaults; 821 of 1827 at 4096, on the 10 evidence
# sessions). 32 x types^2 ran over 5x slower than the defaults; 4096 already
# restores the User heads at 1.4x.
PAIR_CAP = 4096

CORE_ENTITIES = {"User": vt.BASE_ENTITIES["User"], "Person": vt.BASE_ENTITIES["Person"], **vt.VALUE_ENTITIES}
CORE_IDENTITY = {"User": "global", "Person": "global", **dict.fromkeys(vt.VALUE_ENTITIES, "span")}
IDENTITIES = ("global", "chunk", "span")
TYPE_LABEL = re.compile(r"^[A-Z][A-Za-z]{1,30}$")
RELATION_NAME = re.compile(r"^[a-z][a-z_]{1,30}$")

SYSTEM = (
    "You design graph ontologies for an assistant's long-term memory. You answer only with the requested JSON. "
    "The ontology drives a local span extractor (GLiNER2) that reads each conversation turn and emits typed "
    "entities and directed, typed relations between them."
)

PROPOSE_PROMPT = """Below are {n} whole conversations between a user and an AI assistant, sampled across time from one corpus. Propose the domain vocabulary a memory graph for this corpus needs: what should be remembered about the user and their world, so later questions about them can be answered.

A fixed core is always present; do not redefine or rename it:
{core}

Propose, on top of the core:
- entity_types: domain types (CamelCase label, one-line description). The extractor DOES read entity descriptions.
- value_types: only if the corpus needs a value type the core lacks (CamelCase label, description). Usually none.
- relations: directed relations (snake_case name). The extractor sees ONLY THE NAME, never a description, so the name alone must say what the relation means and which way it points. Give intended_head/intended_tail labels (core or proposed) as documentation; the real endpoints are set later from observation. Relations are binary and directed; there are no symmetric or inverse relations.

Prefer the fewest types and relations that cover what this corpus says about its users; every extra label costs the extractor recall on the others. Do not model the assistant, the conversation itself, or generic advice the assistant gives.

Conversations:
{sessions}"""

PRUNE_PROMPT = """You proposed this vocabulary for a memory graph:
{vocab}

A permissive extraction pass (every relation allowed between every pair of types) ran over {n} sessions of the same corpus. Observation tables follow: for each relation, the (head type -> tail type) pairs it actually fired on, with counts and example edges; for each entity type, mention statistics.

Prune the vocabulary:
- For each relation you keep, set head and tail to the endpoint types it should connect, chosen ONLY from pairs it was observed on. Drop pairs that are extraction noise.
- Drop every relation and every non-core type with zero observed instances, and any relation whose observed edges are mostly noise.
- A relation whose head or tail includes User must also include Person there (a mention the model types User may be a third party, re-typed to Person).
- For each kept non-core entity type set identity: "global" (one node per distinct name across all sessions: named things that recur, like people or places), "chunk" (one node per name per session: generic nouns whose same wording in two sessions is not the same thing), or "span" (one node per mention: values). Judge from the statistics, not the label.
- You may not rename anything or add any type or relation.

Core identities are fixed: {core_identity}

Observation:
{tables}"""

PROPOSE_SCHEMA = {
    "type": "object",
    "properties": {
        "entity_types": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"label": {"type": "string"}, "description": {"type": "string"}},
                "required": ["label", "description"],
            },
        },
        "value_types": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"label": {"type": "string"}, "description": {"type": "string"}},
                "required": ["label", "description"],
            },
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "intended_head": {"type": "array", "items": {"type": "string"}},
                    "intended_tail": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["name", "intended_head", "intended_tail"],
            },
        },
    },
    "required": ["entity_types", "value_types", "relations"],
}
PRUNE_SCHEMA = {
    "type": "object",
    "properties": {
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "head": {"type": "array", "items": {"type": "string"}},
                    "tail": {"type": "array", "items": {"type": "string"}},
                    "reason": {"type": "string"},
                },
                "required": ["name", "head", "tail", "reason"],
            },
        },
        "dropped_relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "reason": {"type": "string"}},
                "required": ["name", "reason"],
            },
        },
        "entity_types": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "identity": {"type": "string", "enum": list(IDENTITIES)},
                    "reason": {"type": "string"},
                },
                "required": ["label", "identity", "reason"],
            },
        },
        "dropped_types": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"label": {"type": "string"}, "reason": {"type": "string"}},
                "required": ["label", "reason"],
            },
        },
    },
    "required": ["relations", "dropped_relations", "entity_types", "dropped_types"],
}


def ask(prompt, schema):
    """One headless `claude -p` call; returns (structured output, provenance)."""
    with tempfile.TemporaryDirectory() as cwd:  # no project settings, CLAUDE.md or memory
        started = time.perf_counter()
        done = subprocess.run(
            [
                "claude",
                "-p",
                "--setting-sources",
                "project",
                "--output-format",
                "json",
                "--no-session-persistence",
                "--tools",
                "",
                "--strict-mcp-config",
                "--system-prompt",
                SYSTEM,
                "--json-schema",
                json.dumps(schema),
            ],
            input=prompt,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
    reply = json.loads(done.stdout)
    if reply.get("is_error") or reply.get("structured_output") is None:
        raise RuntimeError(f"LLM call failed: {reply.get('result')!r}")
    usage = reply["usage"]
    return reply["structured_output"], {
        "models": list(reply["modelUsage"]),
        "input_tokens": usage["input_tokens"] + usage["cache_creation_input_tokens"] + usage["cache_read_input_tokens"],
        "output_tokens": usage["output_tokens"],
        "cost_usd": reply["total_cost_usd"],
        "seconds": round(time.perf_counter() - started, 1),
        "prompt_chars": len(prompt),
    }


def excluded_sessions():
    return {s["session_id"] for q in json.loads((HERE / "sample_sessions.json").read_text()) for s in q["sessions"]}


def strata(pool, k):
    """`pool` [(date, sid, chars)] sorted by date, cut into k equal-size time strata."""
    size = len(pool) / k
    return [pool[round(i * size) : round((i + 1) * size)] for i in range(k)]


def samples():
    """Deterministic samples: observe (shared), and disjoint propose samples A and B."""
    excluded = excluded_sessions()
    pool = {}
    for record in vt.stream_records(vt.CORPUS):
        for sid, date, turns in zip(
            record["haystack_session_ids"], record["haystack_dates"], record["haystack_sessions"], strict=True
        ):
            if sid not in excluded and sid not in pool:
                pool[sid] = (date, sum(len(t["content"]) for t in turns))
    ordered = sorted((date, sid, chars) for sid, (date, chars) in pool.items())
    rng = random.Random(SEED)
    observe = [rng.choice(stratum)[1] for stratum in strata(ordered, OBSERVE_SESSIONS)]
    taken, propose = set(observe), {"A": [], "B": []}
    for stratum in strata(ordered, PROPOSE_STRATA):
        a, b = rng.sample([row for row in stratum if row[1] not in taken], 2)
        propose["A"].append(a)
        propose["B"].append(b)
    for name, rows in propose.items():  # whole sessions only: drop the longest until under budget
        while sum(chars for _, _, chars in rows) > PROPOSE_CHARS:
            rows.remove(max(rows, key=lambda row: row[2]))
        propose[name] = [sid for _, sid, _ in rows]
    return observe, propose


def load_texts(sids):
    wanted, docs = set(sids), {}
    for record in vt.stream_records(vt.CORPUS):
        for sid, date, turns in zip(
            record["haystack_session_ids"], record["haystack_dates"], record["haystack_sessions"], strict=True
        ):
            if sid in wanted and sid not in docs:
                docs[sid] = (date, turns)
        if len(docs) == len(wanted):
            break
    return docs


def entity_types(vocab):
    """All entity types the vocabulary declares: the core, then proposed domain and value types."""
    types = dict(CORE_ENTITIES)
    for t in vocab["entity_types"] + vocab["value_types"]:
        types.setdefault(t["label"], t["description"])
    return types


def extract(engine, config, schema, docs):
    """Constrained-by-schema extraction, one turn per window. Returns (edges, mentions)."""
    edges, mentions = [], []
    for done, (sid, (_, turns)) in enumerate(docs.items(), 1):
        print(f"   {done}/{len(docs)} {sid}", flush=True)
        text, spans = turns_of(turns)

        def role(pos, spans=spans):
            return next((r for lo, hi, r in spans if lo <= pos < hi), "?")

        for w in turn_windows(engine, text, spans, pack=False):
            joint = engine.extract(w.text, schema, config=config)
            by_id = {e.id: e for e in joint.entities}
            for e in joint.entities:
                start = e.start + w.start_char
                mentions.append({"sid": sid, "type": e.type, "text": e.text, "start": start, "role": role(start)})
            for r in joint.relations:
                h, t = by_id.get(r.head), by_id.get(r.tail)
                if h is None or t is None:
                    continue
                edges.append(
                    {
                        "sid": sid,
                        "relation": r.type,
                        "head_type": h.type,
                        "head_text": h.text,
                        "head_start": h.start + w.start_char,
                        "tail_type": t.type,
                        "tail_text": t.text,
                        "tail_start": t.start + w.start_char,
                        "confidence": round(r.confidence, 3),
                        "head_role": role(h.start + w.start_char),
                    }
                )
    return edges, mentions


def observation_tables(vocab, edges, mentions):
    """What prune reads: per-relation endpoint pairs with examples, per-type mention statistics."""
    relations = {}
    for rel in vocab["relations"]:
        mine = [e for e in edges if e["relation"] == rel["name"]]
        pairs = Counter((e["head_type"], e["tail_type"]) for e in mine)
        examples = defaultdict(list)
        for e in mine:
            example = f"{e['head_text']} -> {e['tail_text']}"
            bucket = examples[(e["head_type"], e["tail_type"])]
            if example not in bucket and len(bucket) < 4:
                bucket.append(example)
        relations[rel["name"]] = {
            "edges": len(mine),
            "pairs": [
                {"head": h, "tail": t, "count": n, "examples": examples[(h, t)]} for (h, t), n in pairs.most_common(8)
            ],
        }
    types = {}
    for label in entity_types(vocab):
        mine = [m for m in mentions if m["type"] == label]
        texts = Counter(vt.norm(m["text"]) for m in mine)
        sessions_of = defaultdict(set)
        for m in mine:
            sessions_of[vt.norm(m["text"])].add(m["sid"])
        types[label] = {
            "mentions": len(mine),
            "distinct_texts": len(texts),
            "sessions": len({m["sid"] for m in mine}),
            "texts_recurring_across_sessions": round(
                sum(len(s) > 1 for s in sessions_of.values()) / max(len(texts), 1), 2
            ),
            "capitalized_share": round(sum(m["text"][:1].isupper() for m in mine) / max(len(mine), 1), 2),
            "mentions_per_text_per_session": round(len(mine) / max(sum(len(s) for s in sessions_of.values()), 1), 2),
            "top_texts": [t for t, _ in texts.most_common(10)],
        }
    return {"relations": relations, "types": types}


def validate(vocab, pruned, tables):
    """hygm.validation's hard gate, as #347/#353/#355/#358 specified it. Returns the violations."""
    errors = []
    proposed_types = set(entity_types(vocab))
    proposed_relations = {r["name"] for r in vocab["relations"]}
    for label in proposed_types - set(CORE_ENTITIES):
        if not TYPE_LABEL.match(label):
            errors.append(f"type label {label!r} is not CamelCase")
    kept_types = set(CORE_ENTITIES) | {t["label"] for t in pruned["entity_types"]}
    for label in kept_types - proposed_types:
        errors.append(f"type {label!r} was never proposed (prune may not add or rename)")
    for label in {t["label"] for t in pruned["entity_types"]} & set(CORE_ENTITIES):
        errors.append(f"core type {label!r} was re-declared")
    for t in pruned["entity_types"]:
        if t["label"] not in CORE_ENTITIES and tables["types"].get(t["label"], {}).get("mentions", 0) == 0:
            errors.append(f"type {t['label']!r} kept with zero observed mentions")
    for rel in pruned["relations"]:
        name = rel["name"]
        if name not in proposed_relations:
            errors.append(f"relation {name!r} was never proposed")
            continue
        if not RELATION_NAME.match(name):
            errors.append(f"relation name {name!r} is not snake_case")
        observed = {(p["head"], p["tail"]) for p in tables["relations"][name]["pairs"]}
        if not observed:
            errors.append(f"relation {name!r} kept with zero observed instances")
        for side in ("head", "tail"):
            labels = set(rel[side])
            if not labels:
                errors.append(f"relation {name!r} has an empty {side}")
            for label in labels - kept_types:
                errors.append(f"relation {name!r} {side} names undeclared type {label!r}")
            if "User" in labels and "Person" not in labels:
                errors.append(f"relation {name!r}: User in {side} requires Person there too (#358)")
        pairs = {(h, t) for h in rel["head"] for t in rel["tail"]}
        if pairs and not pairs & observed:
            errors.append(f"relation {name!r} keeps no observed endpoint pair")
    return errors


def final_vocabulary(vocab, pruned):
    """The derived ontology as the extractor and the writer consume it."""
    kept = {t["label"]: t["identity"] for t in pruned["entity_types"]}
    types = entity_types(vocab)
    return {
        "entity_types": {
            label: {"description": types[label], "identity": CORE_IDENTITY.get(label) or kept[label]}
            for label in list(CORE_ENTITIES) + [label for label in types if label in kept]
        },
        "relations": [{"name": r["name"], "head": r["head"], "tail": r["tail"]} for r in pruned["relations"]],
    }


def build_schema(engine, types, relations):
    return vt.build_schema(engine, types, [(r["name"], tuple(r["head"]), tuple(r["tail"]), "") for r in relations])


def main(name):
    OUT.mkdir(exist_ok=True)
    path = OUT / f"{name}.json"
    state = json.loads(path.read_text()) if path.exists() else {}

    def save():
        path.write_text(json.dumps(state, indent=1))

    if "samples" not in state:
        observe, propose = samples()
        state["samples"] = {"observe": observe, "propose": propose[name]}
        save()
    print(f"[{name}] propose {len(state['samples']['propose'])} sessions, observe {len(state['samples']['observe'])}")

    if "propose" not in state:
        docs = load_texts(state["samples"]["propose"])
        sessions = "\n\n".join(
            f"=== conversation on {date} ===\n{turns_of(turns)[0]}"
            for date, turns in sorted(docs.values(), key=lambda d: d[0])
        )
        core = "\n".join(f"- {label}: {description}" for label, description in CORE_ENTITIES.items())
        prompt = PROPOSE_PROMPT.format(n=len(docs), core=core, sessions=sessions)
        vocab, provenance = ask(prompt, PROPOSE_SCHEMA)
        state["propose"] = {"vocabulary": vocab, "provenance": provenance}
        save()
    vocab = state["propose"]["vocabulary"]
    print(
        f"[{name}] proposed {len(vocab['entity_types'])} domain types, {len(vocab['value_types'])} value types, {len(vocab['relations'])} relations"
    )
    print(f"[{name}] propose cost {state['propose']['provenance']}")

    if "observe" not in state:
        from gliner2.joint_ie import JointIE, JointIEConfig

        engine = JointIE.from_pretrained(vt.MODEL)
        cap = PAIR_CAP
        config = JointIEConfig(
            include_spans=True, include_confidence=True, relation_pair_cap=cap, max_edges_per_type=cap
        )
        everything = tuple(entity_types(vocab))
        # Held for the whole run (#365).
        schema = build_schema(
            engine,
            entity_types(vocab),
            [{"name": r["name"], "head": everything, "tail": everything} for r in vocab["relations"]],
        )
        docs = load_texts(state["samples"]["observe"])
        started = time.perf_counter()
        edges, mentions = extract(engine, config, schema, docs)
        state["observe"] = {
            "tables": observation_tables(vocab, edges, mentions),
            "seconds": round(time.perf_counter() - started),
            "edges": len(edges),
            "mentions": len(mentions),
            "chars": sum(len(turns_of(t)[0]) for _, t in docs.values()),
            "pair_cap": cap,
        }
        save()
    observed = state["observe"]
    print(
        f"[{name}] observe {observed['edges']} edges, {observed['mentions']} mentions over {observed['chars']} chars in {observed['seconds']}s"
    )

    if "prune" not in state:
        tables = json.dumps(observed["tables"], indent=1)
        prompt = PRUNE_PROMPT.format(
            vocab=json.dumps(vocab, indent=1),
            n=len(state["samples"]["observe"]),
            core_identity=json.dumps(CORE_IDENTITY),
            tables=tables,
        )
        pruned, provenance = ask(prompt, PRUNE_SCHEMA)
        state["prune"] = {"result": pruned, "provenance": provenance}
        save()
    pruned = state["prune"]["result"]
    print(f"[{name}] prune cost {state['prune']['provenance']}")

    errors = validate(vocab, pruned, observed["tables"])
    state["validation"] = errors
    state["vocabulary"] = None if errors else final_vocabulary(vocab, pruned)
    save()
    print(f"[{name}] validation: {'PASS' if not errors else 'FAIL'}")
    for error in errors:
        print(f"   {error}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
