"""OwlImportStrategy: a model imported from a standard OWL ontology (docs/contract.md, section 10)."""

from dataclasses import replace
from pathlib import Path
from typing import Any

from ..errors import ModelError
from ..identifiers import is_valid_identifier
from ..types import HygmModel, NodeType, Property, PropertyType, RelationType
from ..validation import validate_model

_XSD = "http://www.w3.org/2001/XMLSchema#"
_RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"

#: The contract's XSD table (4.6). Any other datatype is a `string`.
_DATATYPES: dict[str, PropertyType] = {
    **{
        _XSD + name: "integer"
        for name in (
            "integer",
            "int",
            "long",
            "short",
            "byte",
            "nonNegativeInteger",
            "nonPositiveInteger",
            "positiveInteger",
            "negativeInteger",
            "unsignedLong",
            "unsignedInt",
            "unsignedShort",
            "unsignedByte",
        )
    },
    _XSD + "float": "float",
    _XSD + "double": "float",
    _XSD + "decimal": "decimal",
    _XSD + "boolean": "boolean",
    _XSD + "date": "date",
    _XSD + "dateTime": "datetime",
    _XSD + "dateTimeStamp": "datetime",
    _RDF + "JSON": "json",
}

#: Cardinality restrictions that say every instance has at least one value.
_AT_LEAST_ONE = ("minCardinality", "cardinality", "minQualifiedCardinality", "qualifiedCardinality")


class OwlImportStrategy:
    """Maps an OWL ontology onto a HygmModel, mechanically:

    - each `owl:Class` (and any class a property's domain or range, or a
      class's `rdfs:subClassOf`, names) is a NodeType, labelled by its IRI's
      local name, described by its `rdfs:comment` (else `rdfs:label`);
    - `rdfs:subClassOf` a named class is a parent. Restrictions, `owl:Thing`
      and a class declared its own subclass are not;
    - each `owl:ObjectProperty` is a RelationType, its `rdfs:domain` and
      `rdfs:range` the start and end labels. A domain or range may be one class
      or an `owl:unionOf` of several; both become a tuple of labels, and a
      missing one means any declared label;
    - each `owl:DatatypeProperty` is a Property on its domain class, or on each
      class of an `owl:unionOf` domain; one with no domain is skipped. Its
      `rdfs:range` is the type by the contract's XSD table, `string` for a
      datatype the table doesn't list, for no range, or for several ranges that
      disagree. A range that is an `owl:oneOf` of literals is a `string`
      property whose `values` are those literals' lexical forms; one that is
      an `owl:onDatatype` restriction takes its base datatype's type;
    - `owl:hasKey` is the class's key, and each key property is required;
    - an `owl:FunctionalProperty` is required on a class that is an
      `rdfs:subClassOf` an `owl:Restriction` on it with `owl:minCardinality`,
      `owl:cardinality`, `owl:minQualifiedCardinality` or
      `owl:qualifiedCardinality` of at least 1;
    - every class, object property and datatype property keeps its IRI as `iri`.

    A key or required property the class only inherits is redeclared on it,
    required. Any other restriction, equivalence, disjointness or axiom is
    ignored: hygm is a schema, not a reasoner. Every node type gets the default
    identity; OWL has no counterpart. Needs the `hygm[owl]` extra (rdflib).
    """

    def create_model(self, path: str | Path, format: str | None = None) -> HygmModel:
        """Parse the ontology at `path` (any serialization rdflib reads; `format` overrides its guess).

        Raises:
            ImportError: if rdflib isn't installed.
            ModelError: (a ValueError) if the file cannot be parsed
                (`invalid_format`), a local name isn't a valid Cypher identifier
                (`invalid_identifier`), a class has more than one `owl:hasKey`
                (`two_keys`), or the model fails validate_model(), carrying its
                codes (a subclass cycle is `parent_cycle`, a key naming
                something other than one of the class's datatype properties
                `unknown_key_property`).
        """
        try:
            from rdflib import Graph
        except ImportError as e:
            raise ImportError("OwlImportStrategy needs rdflib: install `hygm[owl]`") from e

        graph = Graph()
        try:
            graph.parse(str(path), format=format)
        except Exception as e:
            raise ModelError("invalid_format", f"Could not parse OWL ontology {path}: {e}") from e

        model = _Importer(graph, str(path)).model()
        result = validate_model(model)
        if not result.success:
            problems = "; ".join(issue.message for issue in result.critical_issues)
            codes = [issue.code or "invalid_format" for issue in result.critical_issues]
            raise ModelError(codes, f"OWL ontology {path} is not a valid model: {problems}")
        return model


class _Importer:
    """One ontology's mapping; rdflib is imported by the caller before this is built."""

    def __init__(self, graph: Any, source: str) -> None:
        from rdflib.collection import Collection
        from rdflib.namespace import OWL, RDF, RDFS
        from rdflib.term import BNode, Literal, URIRef

        self.graph, self.source = graph, source
        self.Collection, self.OWL, self.RDF, self.RDFS = Collection, OWL, RDF, RDFS
        self.BNode, self.Literal, self.URIRef = BNode, Literal, URIRef

    def model(self) -> HygmModel:
        g, OWL, RDF, RDFS = self.graph, self.OWL, self.RDF, self.RDFS
        object_properties = sorted(set(g.subjects(RDF.type, OWL.ObjectProperty)), key=str)
        datatype_properties = sorted(set(g.subjects(RDF.type, OWL.DatatypeProperty)), key=str)

        declared = {c for c in g.subjects(RDF.type, OWL.Class) if isinstance(c, self.URIRef)}
        referenced = {
            c for p in object_properties for pred in (RDFS.domain, RDFS.range) for c in self.endpoint(p, pred)
        }
        domains = {c for p in datatype_properties for c in self.endpoint(p, RDFS.domain)}
        parents = {c for cls in declared | referenced | domains for c in self.parents(cls)}
        classes = sorted((declared | referenced | domains | parents) - {OWL.Thing}, key=str)

        own: dict[Any, list[Property]] = {c: [] for c in classes}
        for prop in datatype_properties:
            for cls in dict.fromkeys(self.endpoint(prop, RDFS.domain)):
                own[cls].append(self.property(prop))

        node_types = [
            NodeType(
                label=self.local_name(c),
                description=self.text(c),
                parents=tuple(dict.fromkeys(self.local_name(p) for p in self.parents(c))),
                properties=tuple(own[c]),
                key=self.key(c),
                iri=str(c),
            )
            for c in classes
        ]
        relation_types = tuple(
            RelationType(
                label=self.local_name(p),
                description=self.text(p),
                start_labels=tuple(dict.fromkeys(self.local_name(c) for c in self.endpoint(p, RDFS.domain))),
                end_labels=tuple(dict.fromkeys(self.local_name(c) for c in self.endpoint(p, RDFS.range))),
                iri=str(p),
            )
            for p in object_properties
        )
        required = {c: [*self.local_names(self.required_by_restriction(c)), *self.key(c)] for c in classes}
        node_types = self.mark_required(node_types, [required[c] for c in classes])
        return HygmModel(node_types=tuple(node_types), relation_types=relation_types)

    def local_name(self, term: Any) -> str:
        name = str(term)
        name = name.rsplit("#", 1)[-1] if "#" in name else name.rstrip("/").rsplit("/", 1)[-1]
        if not is_valid_identifier(name):
            raise ModelError(
                "invalid_identifier",
                f"OWL ontology {self.source}: {term} has local name {name!r}, not a valid Cypher identifier",
            )
        return name

    def local_names(self, terms: list[Any]) -> list[str]:
        return [self.local_name(t) for t in terms]

    def text(self, term: Any) -> str:
        for predicate in (self.RDFS.comment, self.RDFS.label):
            value = self.graph.value(term, predicate)
            if value is not None:
                return str(value)
        return ""

    def classes_of(self, term: Any) -> list[Any]:
        if isinstance(term, self.URIRef):
            return [term]
        if isinstance(term, self.BNode):
            union = self.graph.value(term, self.OWL.unionOf)
            if union is not None:
                return [m for m in self.Collection(self.graph, union) if isinstance(m, self.URIRef)]
        return []

    def endpoint(self, prop: Any, predicate: Any) -> list[Any]:
        return [cls for value in self.graph.objects(prop, predicate) for cls in self.classes_of(value)]

    def parents(self, cls: Any) -> list[Any]:
        named = (p for p in self.graph.objects(cls, self.RDFS.subClassOf) if isinstance(p, self.URIRef))
        return sorted({p for p in named if p != cls and p != self.OWL.Thing}, key=str)

    def property(self, prop: Any) -> Property:
        kind, values = self.datatype(prop)
        return Property(
            name=self.local_name(prop), type=kind, description=self.text(prop), values=values, iri=str(prop)
        )

    def datatype(self, prop: Any) -> tuple[PropertyType, tuple[str, ...] | None]:
        """The type and allowed values of a datatype property's range."""
        ranges = list(self.graph.objects(prop, self.RDFS.range))
        if len(ranges) == 1 and isinstance(ranges[0], self.BNode):
            one_of = self.graph.value(ranges[0], self.OWL.oneOf)
            if one_of is not None:
                literals = [m for m in self.Collection(self.graph, one_of) if isinstance(m, self.Literal)]
                return "string", tuple(dict.fromkeys(str(m) for m in literals))
        kinds = {self.type_of(r) for r in ranges}
        return (kinds.pop() if len(kinds) == 1 else "string"), None

    def type_of(self, datatype: Any) -> PropertyType:
        if isinstance(datatype, self.BNode):
            datatype = self.graph.value(datatype, self.OWL.onDatatype)
        return _DATATYPES.get(str(datatype), "string")

    def key(self, cls: Any) -> tuple[str, ...]:
        keys = list(self.graph.objects(cls, self.OWL.hasKey))
        if len(keys) > 1:
            raise ModelError(
                "two_keys", f"OWL ontology {self.source}: {cls} has {len(keys)} owl:hasKey axioms; a type has one key"
            )
        return tuple(self.local_names(list(self.Collection(self.graph, keys[0])))) if keys else ()

    def required_by_restriction(self, cls: Any) -> list[Any]:
        """Functional properties a `rdfs:subClassOf` restriction on `cls` says every instance has."""
        g, OWL = self.graph, self.OWL
        found = []
        for restriction in g.objects(cls, self.RDFS.subClassOf):
            if not isinstance(restriction, self.BNode) or (restriction, self.RDF.type, OWL.Restriction) not in g:
                continue
            prop = g.value(restriction, OWL.onProperty)
            if prop is None or (prop, self.RDF.type, OWL.FunctionalProperty) not in g:
                continue
            if any(self.at_least_one(g.value(restriction, OWL[name])) for name in _AT_LEAST_ONE):
                found.append(prop)
        return found

    @staticmethod
    def at_least_one(count: Any) -> bool:
        try:
            return count is not None and int(count) >= 1
        except ValueError:
            return False

    @staticmethod
    def mark_required(node_types: list[NodeType], required: list[list[str]]) -> list[NodeType]:
        """Mark each class's required names; one it only inherits is redeclared on it, required.

        A name the class neither declares nor inherits is left for validate_model() to report.
        """
        partial = HygmModel(node_types=tuple(node_types))
        marked = []
        for node, names in zip(node_types, required, strict=True):
            own = {p.name: p for p in node.properties}
            inherited = {p.name: p for p in partial.properties_of(node.label)}
            for name in names:
                prop = own.get(name) or inherited.get(name)
                if prop is not None:
                    own[name] = replace(prop, required=True)
            marked.append(replace(node, properties=tuple(own.values())))
        return marked
