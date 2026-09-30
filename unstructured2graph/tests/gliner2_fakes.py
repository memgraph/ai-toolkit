"""A fake gliner2 joint engine: no `gliner2` install or model download needed.

It "extracts" by finding known surfaces in each window's text, and emits a
relation only when both its surfaces fall in the same window -- the same
per-window scoping the real engine has.
"""

from types import SimpleNamespace


class FakeSchema:
    def __init__(self):
        self.entities: list[tuple[str, str | None]] = []
        self.relations: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []

    def entity(self, name, description=None):
        self.entities.append((name, description))
        return self

    def relation(self, name, head, tail):
        self.relations.append((name, tuple(head), tuple(tail)))
        return self


class FakeEngine:
    """`surfaces` maps a literal surface to its entity type (case-sensitive, every
    occurrence); `relations` lists (type, head surface, tail surface, confidence)."""

    def __init__(self, surfaces=None, relations=(), feasible=True):
        self.surfaces = dict(surfaces or {})
        self.relations = list(relations)
        self.feasible = feasible
        self.compiled: list[FakeSchema] = []
        self.calls: list[tuple[str, object, object]] = []

    def create_schema(self):
        return FakeSchema()

    def compile_schema(self, schema):
        self.compiled.append(schema)
        return ("compiled", schema)

    def extract(self, text, schema, config=None):
        self.calls.append((text, schema, config))
        entities, first = [], {}
        for surface, entity_type in self.surfaces.items():
            start = text.find(surface)
            while start != -1:
                entity = SimpleNamespace(
                    id=f"e{len(entities)}",
                    type=entity_type,
                    text=surface,
                    start=start,
                    end=start + len(surface),
                    confidence=0.9,
                )
                entities.append(entity)
                first.setdefault(surface, entity.id)
                start = text.find(surface, start + 1)
        relations = [
            SimpleNamespace(type=kind, head=first[head], tail=first[tail], confidence=confidence)
            for kind, head, tail, confidence in self.relations
            if head in first and tail in first
        ]
        return SimpleNamespace(entities=entities, relations=relations, feasible=self.feasible)
