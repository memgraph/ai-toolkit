# Never reject entities for failing ontology conformance

Obvious design: filter/reject entities whose entity_type doesn't match ontology. Deliberately don't. LightRAG's own KV/vector/doc-status stores key off same entity_id, expect the graph node to keep existing — deleting a node post-hoc would desync LightRAG's internal bookkeeping. More fundamental: ontology expected to evolve. Treating it as a lens not a filter means raw entity_type signal always preserved — a later ontology change can retroactively promote labels for entities that didn't conform under an older version, no re-running extraction. Non-conforming entities instead stamped `ontology_conformant: false`: kept, visible, queryable — not silently dropped or indistinguishable from an unprocessed node.

**Considered**: rejecting/deleting non-conforming entities outright — rejected: LightRAG storage-consistency risk above, plus destroys raw signal needed for future re-projection.
