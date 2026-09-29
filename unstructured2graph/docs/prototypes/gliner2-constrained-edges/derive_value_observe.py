"""#386: re-observe relations into core value types with their intended tail.

#372 lost `25:50` in both seeds. Under the permissive observe pass, relations
proposed into a value type never fired into one, so prune dropped them.
This keeps each derivation's samples, proposals and consolidated vocabulary.
It re-observes only the relations whose intended_tail is all core value
types, with that tail and a permissive head. Their permissive tables are
replaced, then prune, validate and the read run again as derivation <name>V.

The re-observe schema holds every entity type, so typing is as before, but
only these relations. In production they decode jointly with all the others,
so this can overstate them slightly; the read on the evidence sessions uses
the full pruned vocabulary.

Observe is sharded over WORKERS processes (one core each is how #372 spent
3.7 h per derivation).

    python derive_value_observe.py S1 S2   # writes derivation/S1V.json, S2V.json
    python read_derived.py S1V S2V
"""

import json
import os
import sys
import time
from multiprocessing import get_context

import derive
import derive_batched
import value_types as vt
from windowing import turns_of

WORKERS = int(os.environ.get("WORKERS", "4"))
VALUE_TYPES = set(vt.VALUE_ENTITIES)


def value_relations(vocab):
    return [r for r in vocab["relations"] if r["intended_tail"] and set(r["intended_tail"]) <= VALUE_TYPES]


def _observe_shard(args):
    vocab, sids, threads = args
    import torch
    from gliner2.joint_ie import JointIE, JointIEConfig

    torch.set_num_threads(threads)
    engine = JointIE.from_pretrained(vt.MODEL)
    config = JointIEConfig(
        include_spans=True,
        include_confidence=True,
        relation_pair_cap=derive.PAIR_CAP,
        max_edges_per_type=derive.PAIR_CAP,
    )
    types = derive.entity_types(vocab)
    everything = tuple(types)
    schema = derive.build_schema(  # held for the whole shard (#365)
        engine,
        types,
        [{"name": r["name"], "head": everything, "tail": tuple(r["intended_tail"])} for r in value_relations(vocab)],
    )
    return derive.extract(engine, config, schema, derive.load_texts(sids))


def main(names):
    for name in names:
        source = json.loads((derive.OUT / f"{name}.json").read_text())
        path = derive.OUT / f"{name}V.json"
        state = (
            json.loads(path.read_text())
            if path.exists()
            else {key: source[key] for key in ("samples", "proposals", "propose")}
        )

        def save(state=state, path=path):
            path.write_text(json.dumps(state, indent=1))

        vocab = state["propose"]["vocabulary"]
        chosen = value_relations(vocab)
        print(f"[{name}V] re-observing {len(chosen)} value relations: {[r['name'] for r in chosen]}", flush=True)

        if "observe" not in state:
            sids = source["samples"]["observe"]
            shards = [sids[i::WORKERS] for i in range(WORKERS)]
            threads = max(1, (os.cpu_count() or WORKERS) // WORKERS)
            started = time.perf_counter()
            with get_context("spawn").Pool(WORKERS) as pool:
                results = pool.map(_observe_shard, [(vocab, shard, threads) for shard in shards])
            edges = [e for shard_edges, _ in results for e in shard_edges]
            mentions = [m for _, shard_mentions in results for m in shard_mentions]
            fresh = derive.observation_tables({**vocab, "relations": chosen}, edges, mentions)
            tables = json.loads(json.dumps(source["observe"]["tables"]))
            tables["relations"].update(fresh["relations"])
            state["observe"] = {
                "tables": tables,
                "value_relations_reobserved": [r["name"] for r in chosen],
                "seconds": round(time.perf_counter() - started),
                "edges": len(edges),
                "workers": WORKERS,
                "chars": sum(len(turns_of(t)[0]) for _, t in derive.load_texts(sids).values()),
            }
            save()
        observed = state["observe"]
        into_values = {
            r["name"]: sum(
                p["count"] for p in observed["tables"]["relations"][r["name"]]["pairs"] if p["tail"] in VALUE_TYPES
            )
            for r in chosen
        }
        print(
            f"[{name}V] observe {observed['edges']} edges in {observed['seconds']}s; into values {into_values}",
            flush=True,
        )

        if "prune" not in state:
            prompt = derive.PRUNE_PROMPT.format(
                vocab=json.dumps(vocab, indent=1),
                n=len(state["samples"]["observe"]),
                core_identity=json.dumps(derive.CORE_IDENTITY),
                tables=json.dumps(observed["tables"], indent=1),
            )
            pruned, provenance = derive.ask(prompt, derive.PRUNE_SCHEMA)
            state["prune"] = {"result": pruned, "provenance": provenance}
            save()
        pruned = state["prune"]["result"]
        errors = derive.validate(vocab, pruned, observed["tables"])
        state["validation"] = errors
        state["vocabulary"] = None if errors else derive.final_vocabulary(vocab, pruned)
        state["unanchored_value_types"] = (
            derive_batched.unanchored_value_types(state["vocabulary"]) if state["vocabulary"] else None
        )
        save()
        kept = [r["name"] for r in pruned["relations"] if r["name"] in {c["name"] for c in chosen}]
        print(f"[{name}V] validation: {'PASS' if not errors else 'FAIL'} {errors}")
        print(f"[{name}V] value relations kept by prune: {kept}")
        print(f"[{name}V] prune cost ${state['prune']['provenance']['cost_usd']:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
