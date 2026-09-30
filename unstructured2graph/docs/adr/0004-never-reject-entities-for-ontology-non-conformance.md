# Never reject entities for failing ontology conformance

Obvious design: filter/reject entities whose entity_type doesn't match ontology. Deliberately don't. For LightRAG specifically, its own KV/vector/doc-status stores key off same entity_id, expect the graph node to keep existing — deleting a node post-hoc would desync LightRAG's internal bookkeeping (`GLiNER2Backend` has no separate storage layer, so this particular risk doesn't apply there). More fundamental, and true regardless of backend: ontology expected to evolve. Treating it as a lens not a filter means raw entity_type signal always preserved — a later ontology change can retroactively promote labels for entities that didn't conform under an older version, no re-running extraction. Non-conforming entities instead stamped `ontology_conformant: false`: kept, visible, queryable — not silently dropped or indistinguishable from an unprocessed node.

**Considered**: rejecting/deleting non-conforming entities outright — rejected: LightRAG storage-consistency risk above, plus destroys raw signal needed for future re-projection (this second reason alone would justify the policy even for a backend without LightRAG's storage risk).

## Amendment: relations, and the two compilations (map #344)

The policy covers relationships too. A relationship of a declared type whose endpoints fall outside its `start_labels`/`end_labels` is kept and stamped `r.ontology_conformant = false` by `enforce_relation_domain_range`, and a rerun clears the flag on one that now conforms. It is never deleted.

One specification, `start_labels`/`end_labels`, is compiled twice:
- **At extraction**, to GLiNER2's `JointSchema` typing. Its job is quality: steering the decoder. It is GLiNER2-only.
- **Post hoc**, to a Cypher check over Memgraph. Its job is integrity. It runs on every backend, is authoritative, and is gated by `enforce_ontology`.

The same spec goes into both, with no deliberate loosening, which buys an invariant: on GLiNER2 output a post-hoc flag means a bug. Domain/range survives the window merge, and entity types are the ontology's own list. So the flags are an **integrity alarm, not a read filter**. Their counts go into `ReconciliationSummary`, and readers never filter on them (#355).

The cost this ADR's re-projection argument warns about is real for relations. The extraction-time half is decode-time suppression: GLiNER2 never emits an edge its declared range excludes, so a later, wider ontology cannot re-project it without re-extracting. #350 measured a real case: `visited → Museum of Modern Art` was lost to a range one label too narrow. Two things mitigate it:
- The range comes from observation, so it follows the model's typing.
- A permissive shadow sample reports suppressed endpoint pairs as widening candidates (#359). This is reporting only, and not built yet.

Entity identity is the opposite case: merging mentions into one node (#346) destroys nothing, because every mention keeps its `MENTIONED_IN` edge.

**Considered**: loosening the extraction-time constraint relative to the post-hoc one, to reduce suppression. Rejected, because it forfeits the bug invariant above for a benefit the shadow sample delivers without it.
