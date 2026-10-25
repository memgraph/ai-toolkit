# The eval runs named workloads through one pipeline, and keeps each workload's graph

Architecture review #549 (item 5) found the eval branching three retrieval strategies through its runner, report and CLI, with BEAM copying LongMemEval's staging and importing a private function to do it. It also found no line between the numbers we publish and the runs we iterate on: both were the same `run`/`beam` commands with different flags, and every run wiped and rebuilt its graph, so iterating on recall paid hours of reconstruction each time.

## Decision

- **Recall is the only thing measured.** The graph-agent and text-search strategies are deleted. Neither is the frontier, and neither will be run again. LightRAG extraction and the hand-written LongMemEval vocabulary (`--ontology fixed`) go with them: a build is GLiNER2 under hygm's default model, or a learned one, which is what the product ships.
- **One pipeline, benchmark adapters.** `pipeline.py` builds (inject, reconcile, derive, embed) and asks (recall, answer). A benchmark adapter (`benchmarks/`) brings only its questions, haystack, judge and headline. Every adapter reduces a question to one score from 0 to 1, so the run format, the summary, the comparison and the noise floor are written once.
- **Named workloads, official and dev.** What runs is a workload from a fixed registry (`workloads.py`). Official workloads are whole benchmarks and back the published numbers. Dev workloads are fixed slices (LongMemEval-S: 50 questions, 10-session cap; BEAM 100K: 4 chats) for comparing two versions of the system, never for quoting.
- **Each workload keeps its graph.** It has its own named, durable Memgraph container. A manifest fingerprints what built the graph: the questions, the build settings, and the source of every module that shapes it. A run reuses a matching graph, resumes an interrupted one, and rebuilds otherwise, so nobody has to decide whether a change needs a rebuild.

## Consequences

- A recall change re-asks a dev workload in minutes; a construction change rebuilds it automatically. A change to a module listed in `pipeline._CONSTRUCTION_MODULES` that doesn't change the graph still rebuilds; that list errs toward rebuilding.
- The official LongMemEval-S run is sharded by the workload itself (25 of 20), and resumes from saved shards.
- Published baselines and local runs share one format; `publish` strips benchmark text. The two existing baselines were converted, with their headlines unchanged.
- Saved runs of the deleted strategies don't load. Nothing in the repo refers to them, and the context-graph family has no users yet who would need them.
