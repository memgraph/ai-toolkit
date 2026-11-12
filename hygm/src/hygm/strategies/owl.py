"""OwlImportStrategy: a model imported from a standard OWL ontology (docs/contract.md, section 10).

The mapping is mechanical, construct by construct; `owl_export.model_to_owl`
writes the same table backwards. What OWL has no construct for travels as an
annotation in the `hygm:` namespace (`HYGM`), which other OWL tools carry
along and ignore.
"""

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from ..errors import ModelError
from ..format import check_profile, property_from_mapping
from ..identifiers import is_valid_identifier
from ..types import (
    Conventions,
    HygmModel,
    Import,
    Individual,
    NodeType,
    Ontology,
    Property,
    PropertyType,
    RelationRestriction,
    RelationType,
)
from ..validation import validate_model

_XSD = "http://www.w3.org/2001/XMLSchema#"
_RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"

#: The namespace of hygm's own annotations: what a model says that OWL has no construct for.
HYGM = "https://memgraph.com/ns/hygm#"

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

#: Cardinality restrictions giving the fewest values, and those giving the most.
_LEAST = ("minCardinality", "cardinality", "minQualifiedCardinality", "qualifiedCardinality")
_MOST = ("maxCardinality", "cardinality", "maxQualifiedCardinality", "qualifiedCardinality")

#: Files an `owl:imports` may name that are hygm model files rather than OWL.
_MODEL_FILE_SUFFIXES = (".yaml", ".yml", ".json")


class OwlImportStrategy:
    """Maps an OWL ontology onto a HygmModel, mechanically (docs/contract.md, section 10):

    - each `owl:Class` (and any class a property's domain or range, or a
      class's `rdfs:subClassOf`, names) is a NodeType, labelled by its IRI's
      local name, described by its `rdfs:comment` (else an `rdfs:label`);
    - `rdfs:subClassOf` a named class is a parent; `owl:Thing` and a class
      declared its own subclass are not. `owl:disjointWith` and
      `owl:AllDisjointClasses` are `disjoint_with`;
    - each `owl:ObjectProperty` is a RelationType, its `rdfs:domain` and
      `rdfs:range` the start and end labels. A domain or range may be one class
      or an `owl:unionOf` of several; a missing one means any declared label,
      and so do `owl:Thing` and a union containing it. `owl:inverseOf`,
      `owl:SymmetricProperty` and `owl:FunctionalProperty` (`max_count: 1`)
      are its facets;
    - each `owl:DatatypeProperty` is a Property on its domain class, or on each
      class of an `owl:unionOf` domain; one with no domain, or whose domain is
      `owl:Thing` or a union containing it, is skipped. Its `rdfs:range` is
      the type by the contract's XSD table, `string` for a datatype the table
      doesn't list, for no range, or for several ranges that disagree. A range
      that is an `owl:oneOf` of literals gives `values`; an `owl:onDatatype`
      restriction gives its base datatype's type, and its `xsd:minInclusive`
      and `xsd:maxInclusive` the bounds. It is single-valued unless it says
      otherwise (`hygm:multiValued`, or a larger `owl:maxCardinality`);
    - a restriction a class is an `rdfs:subClassOf` sets, on its own datatype
      property, the counts (`min`/`max`/exact cardinality, qualified or not)
      and `value` (`owl:hasValue`). On an inherited one it redeclares the
      property there, narrowed, `owl:allValuesFrom` a data range included. On
      an object property it is a relation restriction: the counts, and the end
      labels of `owl:allValuesFrom`;
    - `owl:hasKey` is the class's key, and each key property is required;
    - language-tagged `rdfs:label` and `skos:prefLabel` are `names`, and
      `skos:altLabel` `synonyms`;
    - `owl:NamedIndividual` is an individual of its class, its data and
      object property assertions its properties and relations;
    - the `owl:Ontology` is the `ontology` block: its `rdfs:label`,
      `rdfs:comment` and `owl:versionInfo`. Its `owl:imports` of a local file
      is an import, converted too (a hygm model file or OWL); one of a web IRI
      is left out, and the classes it declares are only those the ontology
      names;
    - everything keeps its IRI as `iri`, except one under the `hygm:base` an
      export minted it from.

    A key or required property the class only inherits is redeclared on it,
    required. Equivalence and other axioms are ignored: hygm is a schema, not
    a reasoner. OWL keeps no order, so the model lists types, properties and
    individuals in IRI order and the values of a multi-valued annotation
    sorted. Needs the `hygm[owl]` extra (rdflib).
    """

    def create_model(self, path: str | Path, format: str | None = None) -> HygmModel:
        """Parse the ontology at `path` (any serialization rdflib reads; `format` overrides its guess).

        Raises:
            ImportError: if rdflib isn't installed.
            ModelError: (a ValueError) if the file cannot be parsed
                (`invalid_format`), a local name isn't a valid Cypher identifier
                (`invalid_identifier`), a class has more than one `owl:hasKey`
                (`two_keys`), an import is missing (`unknown_model`) or forms a
                cycle (`import_cycle`), or the model fails validate_model(),
                carrying its codes (a subclass cycle is `parent_cycle`, a key
                naming something other than one of the class's datatype
                properties `unknown_key_property`).
        """
        return _read(source=Path(path), data=None, format=format, importing=())


def model_from_owl(data: str, *, format: str = "turtle", location: str | Path | None = None) -> HygmModel:
    """A model from an OWL document held in memory, as OwlImportStrategy reads a file.

    Args:
        data: The document.
        format: Its serialization, as rdflib names it.
        location: Where the document notionally lives: relative IRIs and
            `owl:imports` resolve against it. None resolves nothing relative.

    Raises:
        ModelError: as OwlImportStrategy.create_model().
    """
    return _read(source=Path(location) if location else None, data=data, format=format, importing=())


def _read(source: Path | None, data: str | None, format: str | None, importing: tuple[Path, ...]) -> HygmModel:
    try:
        from rdflib import Graph
    except ImportError as e:
        raise ImportError("OWL import needs rdflib: install `hygm[owl]`") from e

    import rdflib

    where = str(source) if source else "OWL document"
    graph = Graph()
    # hygm keeps dates as the ISO 8601 text they were written in; rdflib's
    # parser would otherwise rewrite each literal's lexical form (`Z` to
    # `+00:00`). The switch is process-wide, so it is restored right after.
    normalize = rdflib.NORMALIZE_LITERALS
    setattr(rdflib, "NORMALIZE_LITERALS", False)  # noqa: B010  (rdflib declares the flag as the constant True)
    try:
        if data is None:
            graph.parse(str(source), format=format)
        else:
            graph.parse(data=data, format=format, publicID=source.resolve().as_uri() if source else None)
    except Exception as e:
        raise ModelError("invalid_format", f"Could not parse OWL ontology {where}: {e}") from e
    finally:
        setattr(rdflib, "NORMALIZE_LITERALS", normalize)  # noqa: B010

    model = _Importer(graph, where, source, importing).model()
    result = validate_model(model)
    if not result.success:
        problems = "; ".join(issue.message for issue in result.critical_issues)
        codes = [issue.code or "invalid_format" for issue in result.critical_issues]
        raise ModelError(codes, f"OWL ontology {where} is not a valid model: {problems}")
    return model


def json_literal(value: Any) -> str:
    """`value` as the JSON text of a `hygm:` annotation."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


class _Importer:
    """One ontology's mapping; rdflib is imported by the caller before this is built."""

    def __init__(self, graph: Any, where: str, source: Path | None, importing: tuple[Path, ...]) -> None:
        from rdflib import Namespace
        from rdflib.collection import Collection
        from rdflib.namespace import OWL, RDF, RDFS, SKOS, XSD
        from rdflib.term import BNode, Literal, URIRef

        self.graph, self.where, self.source, self.importing = graph, where, source, importing
        self.Collection, self.OWL, self.RDF, self.RDFS, self.SKOS, self.XSD = Collection, OWL, RDF, RDFS, SKOS, XSD
        self.BNode, self.Literal, self.URIRef = BNode, Literal, URIRef
        self.HYGM = Namespace(HYGM)
        headers = sorted(graph.subjects(RDF.type, OWL.Ontology), key=str)
        self.header = headers[0] if headers else None
        base = graph.value(self.header, self.HYGM.base) if self.header is not None else None
        self.base = str(base) if base is not None else None

    def model(self) -> HygmModel:
        g, OWL, RDF, RDFS = self.graph, self.OWL, self.RDF, self.RDFS
        imports = self.imports()
        imported = {t.label for i in imports for t in i.model.node_types}
        object_properties = sorted(set(g.subjects(RDF.type, OWL.ObjectProperty)), key=str)
        datatype_properties = sorted(set(g.subjects(RDF.type, OWL.DatatypeProperty)), key=str)
        self.datatype_properties = set(datatype_properties)

        declared = {c for c in g.subjects(RDF.type, OWL.Class) if isinstance(c, self.URIRef)}
        referenced = {
            c for p in object_properties for pred in (RDFS.domain, RDFS.range) for c in self.endpoint(p, pred)
        }
        domains = {c for p in datatype_properties for c in self.endpoint(p, RDFS.domain)}
        parents = {c for cls in declared | referenced | domains for c in self.parents(cls)}
        classes = sorted(
            c
            for c in (declared | referenced | domains | parents) - {OWL.Thing}
            if c in declared or self.label(c) not in imported
        )

        own: dict[Any, list[Property]] = {c: [] for c in classes}
        for prop in datatype_properties:
            for cls in dict.fromkeys(self.endpoint(prop, RDFS.domain)):
                if cls in own:
                    own[cls].append(self.property(prop, cls))

        node_types = [
            NodeType(
                label=self.label(c),
                description=self.text(c),
                parents=tuple(dict.fromkeys(self.label(p) for p in self.parents(c))),
                properties=tuple(own[c]),
                key=self.key(c),
                iri=self.iri(c),
                profiles=self.profiles(c, "node"),
                abstract=self.flag(c, self.HYGM.abstract),
                disjoint_with=self.disjoint(c),
                synonyms=self.synonyms(c),
                names=self.names(c),
                relation_restrictions=self.relation_restrictions(c),
            )
            for c in classes
        ]
        node_types = self.narrow_inherited(node_types, classes, imports)
        relation_types = tuple(self.relation(p) for p in object_properties)
        required = [self.key(c) for c in classes]
        node_types = self.mark_required(node_types, required, imports)
        partial = HygmModel(
            node_types=_merged(imports, "node_types", tuple(node_types)),
            relation_types=_merged(imports, "relation_types", relation_types),
        )
        return replace(
            partial,
            profiles=self.profiles(self.header, "model") if self.header is not None else {},
            ontology=self.ontology(),
            imports=imports,
            individuals=_merged(imports, "individuals", self.individuals(partial)),
        )

    # Names and IRIs

    def local_name(self, term: Any) -> str:
        name = str(term)
        name = name.rsplit("#", 1)[-1] if "#" in name else name.rstrip("/").rsplit("/", 1)[-1]
        return self.identifier(name, term)

    def identifier(self, name: str, term: Any) -> str:
        if not is_valid_identifier(name):
            raise ModelError(
                "invalid_identifier",
                f"OWL ontology {self.where}: {term} has local name {name!r}, not a valid Cypher identifier",
            )
        return name

    def label(self, term: Any) -> str:
        """`term`'s hygm name: its `hygm:name`, else its local name."""
        name = self.graph.value(term, self.HYGM.name)
        return self.identifier(str(name), term) if name is not None else self.local_name(term)

    def labels(self, terms: list[Any]) -> list[str]:
        return [self.label(t) for t in terms]

    def iri(self, term: Any) -> str | None:
        """`term`'s IRI, or None for one an export minted under its `hygm:base`."""
        return None if self.base is not None and str(term).startswith(self.base) else str(term)

    # Annotations

    def text(self, term: Any) -> str:
        comment = self.graph.value(term, self.RDFS.comment)
        if comment is not None:
            return str(comment)
        labels = sorted(self.graph.objects(term, self.RDFS.label), key=lambda label: (bool(label.language), label))
        return str(labels[0]) if labels else ""

    def names(self, term: Any) -> dict[str, str]:
        names: dict[str, str] = {}
        for predicate in (self.RDFS.label, self.SKOS.prefLabel):
            for label in sorted(self.graph.objects(term, predicate)):
                if isinstance(label, self.Literal) and label.language:
                    names.setdefault(label.language, str(label))
        return names

    def strings(self, term: Any, predicate: Any) -> tuple[str, ...]:
        return tuple(sorted(str(value) for value in self.graph.objects(term, predicate)))

    def synonyms(self, term: Any) -> tuple[str, ...]:
        return self.strings(term, self.SKOS.altLabel)

    def flag(self, term: Any, predicate: Any) -> bool:
        value = self.graph.value(term, predicate)
        return bool(value.toPython()) if value is not None else False

    def json(self, term: Any, predicate: Any) -> Any:
        value = self.graph.value(term, predicate)
        if value is None:
            return None
        try:
            return json.loads(str(value))
        except ValueError as e:
            raise ModelError("invalid_format", f"OWL ontology {self.where}: {term} {predicate} isn't JSON") from e

    def count(self, term: Any, predicate: Any) -> int | None:
        value = self.graph.value(term, predicate)
        return _whole(value)

    def profiles(self, term: Any, level: str) -> dict[str, Any]:
        profiles = self.json(term, self.HYGM.profiles) or {}
        for name, settings in profiles.items():
            check_profile(name, settings, level, f"OWL ontology {self.where} {term}")
        return profiles

    # Classes

    def classes_of(self, term: Any) -> list[Any]:
        """The named classes `term` names: none for `owl:Thing` or a union containing it, which mean anything."""
        members = [term]
        if isinstance(term, self.BNode):
            union = self.graph.value(term, self.OWL.unionOf)
            members = list(self.Collection(self.graph, union)) if union is not None else []
        if self.OWL.Thing in members:
            return []
        return [m for m in members if isinstance(m, self.URIRef)]

    def endpoint(self, prop: Any, predicate: Any) -> list[Any]:
        return [cls for value in self.graph.objects(prop, predicate) for cls in self.classes_of(value)]

    def parents(self, cls: Any) -> list[Any]:
        named = (p for p in self.graph.objects(cls, self.RDFS.subClassOf) if isinstance(p, self.URIRef))
        return sorted({p for p in named if p != cls and p != self.OWL.Thing}, key=str)

    def disjoint(self, cls: Any) -> tuple[str, ...]:
        """The classes declared disjoint with `cls`, each pair once: on its subject, or the earlier member of a list."""
        g, OWL = self.graph, self.OWL
        found = [d for d in g.objects(cls, OWL.disjointWith) if isinstance(d, self.URIRef)]
        for axiom in g.subjects(self.RDF.type, OWL.AllDisjointClasses):
            members = list(self.Collection(g, g.value(axiom, OWL.members)))
            if cls in members:
                found += [m for m in members[members.index(cls) + 1 :] if isinstance(m, self.URIRef)]
        return tuple(sorted(dict.fromkeys(self.label(d) for d in found if d != cls)))

    def restrictions(self, cls: Any) -> list[Any]:
        g = self.graph
        return [
            r
            for r in g.objects(cls, self.RDFS.subClassOf)
            if isinstance(r, self.BNode) and (r, self.RDF.type, self.OWL.Restriction) in g
        ]

    def counts(self, restriction: Any) -> tuple[int | None, int | None]:
        least = [n for name in _LEAST if (n := self.count(restriction, self.OWL[name])) is not None]
        most = [n for name in _MOST if (n := self.count(restriction, self.OWL[name])) is not None]
        return (max(least) if least else None), (min(most) if most else None)

    def relation_restrictions(self, cls: Any) -> tuple[RelationRestriction, ...]:
        found: dict[str, dict[str, Any]] = {}
        for restriction in self.restrictions(cls):
            prop = self.graph.value(restriction, self.OWL.onProperty)
            if prop is None or (prop, self.RDF.type, self.OWL.ObjectProperty) not in self.graph:
                continue
            entry = found.setdefault(self.label(prop), {})
            least, most = self.counts(restriction)
            if least is not None:
                entry["min_count"] = max(least, entry.get("min_count", least))
            if most is not None:
                entry["max_count"] = min(most, entry.get("max_count", most))
            every = self.graph.value(restriction, self.OWL.allValuesFrom)
            if every is not None:
                entry["end_labels"] = tuple(sorted(self.labels(self.classes_of(every))))
        return tuple(RelationRestriction(relation, **entry) for relation, entry in sorted(found.items()))

    def key(self, cls: Any) -> tuple[str, ...]:
        keys = list(self.graph.objects(cls, self.OWL.hasKey))
        if len(keys) > 1:
            raise ModelError(
                "two_keys", f"OWL ontology {self.where}: {cls} has {len(keys)} owl:hasKey axioms; a type has one key"
            )
        return tuple(self.labels(list(self.Collection(self.graph, keys[0])))) if keys else ()

    # Datatype properties

    def property(self, prop: Any, cls: Any) -> Property:
        """Datatype property `prop` as class `cls` declares it, with the facets its restrictions there set."""
        kind, values, minimum, maximum = self.datatype(prop)
        functional = (prop, self.RDF.type, self.OWL.FunctionalProperty) in self.graph
        fields: dict[str, Any] = {
            "max_count": None if self.flag(prop, self.HYGM.multiValued) else 1,
            "min_count": None,
        }
        if functional:
            fields["max_count"] = 1
        fields.update(self.data_facets(cls, prop, kind, multi_valued=fields["max_count"] != 1))
        default = self.json(prop, self.HYGM.default)
        unit = self.graph.value(prop, self.HYGM.unit)
        return Property(
            name=self.label(prop),
            type=kind,
            description=self.text(prop),
            values=values,
            iri=self.iri(prop),
            profiles=self.profiles(prop, "property"),
            synonyms=self.synonyms(prop),
            names=self.names(prop),
            default=default,
            minimum=minimum,
            maximum=maximum,
            unit=str(unit) if unit is not None else None,
            **fields,
        )

    def data_facets(self, cls: Any, prop: Any, kind: PropertyType, *, multi_valued: bool) -> dict[str, Any]:
        """The counts and `value` the restrictions on `cls` set for datatype property `prop`."""
        facets: dict[str, Any] = {}
        values = []
        for restriction in self.restrictions(cls):
            if self.graph.value(restriction, self.OWL.onProperty) != prop:
                continue
            least, most = self.counts(restriction)
            if least is not None:
                facets["min_count"] = least
            if most is not None:
                facets["max_count"] = most
                multi_valued = most != 1
            value = self.graph.value(restriction, self.OWL.hasValue)
            if value is not None:
                values.append(_value(value, kind))
        if values:
            facets["value"] = sorted(values, key=json_literal) if multi_valued else values[0]
        return facets

    def datatype(self, prop: Any) -> tuple[PropertyType, tuple[str, ...] | None, Any, Any]:
        """The type, allowed values and bounds of a datatype property's range."""
        ranges = list(self.graph.objects(prop, self.RDFS.range))
        if len(ranges) == 1 and isinstance(ranges[0], self.BNode):
            one_of = self.graph.value(ranges[0], self.OWL.oneOf)
            if one_of is not None:
                literals = [m for m in self.Collection(self.graph, one_of) if isinstance(m, self.Literal)]
                return "string", tuple(dict.fromkeys(str(m) for m in literals)), None, None
        kinds = {self.type_of(r) for r in ranges}
        kind: PropertyType = kinds.pop() if len(kinds) == 1 else "string"
        minimum, maximum = self.bounds(ranges[0], kind) if len(ranges) == 1 else (None, None)
        return kind, None, minimum, maximum

    def type_of(self, datatype: Any) -> PropertyType:
        if isinstance(datatype, self.BNode):
            datatype = self.graph.value(datatype, self.OWL.onDatatype)
        return _DATATYPES.get(str(datatype), "string")

    def bounds(self, datatype: Any, kind: PropertyType) -> tuple[Any, Any]:
        """The `xsd:minInclusive` and `xsd:maxInclusive` of an `owl:withRestrictions` data range."""
        facets = self.graph.value(datatype, self.OWL.withRestrictions) if isinstance(datatype, self.BNode) else None
        if facets is None:
            return None, None
        minimum = maximum = None
        for facet in self.Collection(self.graph, facets):
            low = self.graph.value(facet, self.XSD.minInclusive)
            high = self.graph.value(facet, self.XSD.maxInclusive)
            minimum = _value(low, kind) if low is not None else minimum
            maximum = _value(high, kind) if high is not None else maximum
        return minimum, maximum

    def narrow_inherited(
        self, node_types: list[NodeType], classes: list[Any], imports: tuple[Import, ...]
    ) -> list[NodeType]:
        """Redeclare an inherited datatype property, narrowed, on a class whose restriction narrows it there.

        Ancestors first, so a restriction lower down narrows what one higher up left.
        """
        by_label = {t.label: t for t in node_types}
        order = sorted(
            zip(classes, node_types, strict=True),
            key=lambda pair: len(self.ancestry(pair[1].label, by_label, imports)),
        )
        for cls, node in order:
            node = by_label[node.label]
            partial = HygmModel(node_types=(*_merged(imports, "node_types", ()), *by_label.values()))
            own = {p.name: p for p in node.properties}
            for restriction in self.restrictions(cls):
                prop = self.graph.value(restriction, self.OWL.onProperty)
                if prop not in self.datatype_properties:
                    continue
                name = self.label(prop)
                inherited = partial.inherited(node.label, name)
                if name in own or inherited is None:
                    continue
                own[name] = self.narrowed(inherited, restriction)
            by_label[node.label] = replace(node, properties=tuple(own.values()))
        return [by_label[t.label] for t in node_types]

    @staticmethod
    def ancestry(label: str, by_label: dict[str, NodeType], imports: tuple[Import, ...]) -> tuple[str, ...]:
        model = HygmModel(node_types=(*_merged(imports, "node_types", ()), *by_label.values()))
        return model.ancestors(label)

    def narrowed(self, inherited: Property, restriction: Any) -> Property:
        least, most = self.counts(restriction)
        changes: dict[str, Any] = {}
        if least is not None:
            changes.update(required=least >= 1, min_count=least)
        if most is not None:
            changes["max_count"] = most
        value = self.graph.value(restriction, self.OWL.hasValue)
        if value is not None:
            changes["value"] = _value(value, inherited.type)
        every = self.graph.value(restriction, self.OWL.allValuesFrom)
        if every is not None:
            one_of = self.graph.value(every, self.OWL.oneOf) if isinstance(every, self.BNode) else None
            if one_of is not None:
                changes["values"] = tuple(str(m) for m in self.Collection(self.graph, one_of))
            minimum, maximum = self.bounds(every, inherited.type)
            changes.update({k: v for k, v in (("minimum", minimum), ("maximum", maximum)) if v is not None})
        return replace(inherited, **changes)

    # Object properties

    def relation(self, prop: Any) -> RelationType:
        g, OWL, RDFS = self.graph, self.OWL, self.RDFS
        functional = (prop, self.RDF.type, OWL.FunctionalProperty) in g
        properties = self.json(prop, self.HYGM.properties) or []
        inverse = g.value(prop, OWL.inverseOf)
        max_count = self.count(prop, self.HYGM.maxCount)
        return RelationType(
            label=self.label(prop),
            description=self.text(prop),
            start_labels=tuple(dict.fromkeys(self.label(c) for c in self.endpoint(prop, RDFS.domain))),
            end_labels=tuple(dict.fromkeys(self.label(c) for c in self.endpoint(prop, RDFS.range))),
            properties=tuple(
                property_from_mapping(item, f"OWL ontology {self.where} {prop} properties") for item in properties
            ),
            iri=self.iri(prop),
            profiles=self.profiles(prop, "relation"),
            synonyms=self.synonyms(prop),
            names=self.names(prop),
            min_count=self.count(prop, self.HYGM.minCount) or 0,
            max_count=1 if functional else max_count,
            inverse_of=self.label(inverse) if inverse is not None else None,
            symmetric=(prop, self.RDF.type, OWL.SymmetricProperty) in g,
        )

    # Required properties

    @staticmethod
    def mark_required(
        node_types: list[NodeType], required: list[tuple[str, ...]], imports: tuple[Import, ...]
    ) -> list[NodeType]:
        """Mark each class's key properties required; one it only inherits is redeclared on it, required.

        A name the class neither declares nor inherits is left for validate_model() to report.
        """
        partial = HygmModel(node_types=(*_merged(imports, "node_types", ()), *node_types))
        marked = []
        for node, names in zip(node_types, required, strict=True):
            own = {p.name: p for p in node.properties}
            inherited = {p.name: p for p in partial.properties_of(node.label)}
            for name in names:
                prop = own.get(name) or inherited.get(name)
                if prop is not None and not prop.required:
                    own[name] = replace(prop, required=True)
            marked.append(replace(node, properties=tuple(own.values())))
        return marked

    # Individuals

    def individuals(self, model: HygmModel) -> tuple[Individual, ...]:
        g, RDF, OWL = self.graph, self.RDF, self.OWL
        individuals = []
        for subject in sorted(set(g.subjects(RDF.type, OWL.NamedIndividual)), key=str):
            types = sorted(
                label
                for t in g.objects(subject, RDF.type)
                if t != OWL.NamedIndividual and isinstance(t, self.URIRef)
                if model.node_type(label := self.label(t)) is not None
            )
            if not types:
                continue
            effective = {p.name: p for p in model.properties_of(types[0])}
            values: dict[str, list[Any]] = {name: [] for name in self.strings(subject, self.HYGM.noValues)}
            relations: dict[str, list[str]] = {}
            for predicate, value in g.predicate_objects(subject):
                if predicate in self.datatype_properties and isinstance(value, self.Literal):
                    name = self.label(predicate)
                    kind = effective[name].type if name in effective else "string"
                    values.setdefault(name, []).append(_value(value, kind))
                elif (predicate, RDF.type, OWL.ObjectProperty) in g and isinstance(value, self.URIRef):
                    relations.setdefault(self.label(predicate), []).append(self.label(value))
            properties = {
                name: sorted(found, key=json_literal)
                if name in effective and effective[name].multi_valued
                else found[0]
                if found
                else None
                for name, found in sorted(values.items())
            }
            individuals.append(
                Individual(
                    self.label(subject),
                    types[0],
                    iri=self.iri(subject),
                    properties=properties,
                    relations={name: tuple(sorted(targets)) for name, targets in sorted(relations.items())},
                )
            )
        return tuple(individuals)

    # The ontology and its imports

    def ontology(self) -> Ontology | None:
        header, g, HYGM = self.header, self.graph, self.HYGM
        if header is None:
            return None
        minted = self.base is not None and str(header) == self.base.rstrip("#/")
        name = next((str(label) for label in g.objects(header, self.RDFS.label) if not label.language), None)
        comment = g.value(header, self.RDFS.comment)
        version = g.value(header, self.OWL.versionInfo)
        conventions = self.json(header, HYGM.conventions)
        fields = {
            "name": name,
            "description": str(comment) if comment is not None else None,
            "iri": None if minted else str(header),
            "version_info": str(version) if version is not None else None,
            "includes": self.strings(header, HYGM.includes),
            "excludes": self.strings(header, HYGM.excludes),
            "competency_questions": self.strings(header, HYGM.competencyQuestion),
            "maintainers": self.strings(header, HYGM.maintainer),
            "conventions": Conventions(**conventions) if conventions is not None else None,
        }
        if not any(fields.values()) and not self.flag(header, HYGM.ontologyBlock):
            return None
        return Ontology(**fields)

    def imports(self) -> tuple[Import, ...]:
        """Each `owl:imports` of a local file, converted; one of a web IRI is left out."""
        if self.header is None:
            return ()
        found = []
        for target in sorted(self.graph.objects(self.header, self.OWL.imports), key=str):
            path = _local_path(str(target))
            if path is None:
                continue
            if not path.exists():
                raise ModelError("unknown_model", f"OWL ontology {self.where} imports {path}, which doesn't exist")
            if path.resolve() in self.importing or (self.source and path.resolve() == self.source.resolve()):
                raise ModelError("import_cycle", f"OWL ontology {self.where} imports {path}, which imports it back")
            found.append(Import(self.relative(path), self.load(path)))
        return tuple(found)

    def relative(self, path: Path) -> str:
        if self.source is None:
            return str(path)
        return os.path.relpath(path.resolve(), self.source.resolve().parent).replace(os.sep, "/")

    def load(self, path: Path) -> HygmModel:
        if path.suffix in _MODEL_FILE_SUFFIXES:
            from .manual import ManualStrategy

            return ManualStrategy().create_model(path)
        importing = (*self.importing, *((self.source.resolve(),) if self.source else ()))
        return _read(source=path, data=None, format=None, importing=importing)


def _local_path(iri: str) -> Path | None:
    parsed = urlparse(iri)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path))
    if not parsed.scheme:
        return Path(iri)
    return None


def _whole(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _value(literal: Any, kind: PropertyType) -> Any:
    """A literal as the JSON value a property of type `kind` holds: ISO 8601 text for dates (4.14)."""
    if kind == "json":
        return json.loads(str(literal))
    if kind in ("string", "date", "datetime"):
        return str(literal)
    value = literal.toPython()
    if kind == "boolean":
        return bool(value) if isinstance(value, bool) else str(literal) == "true"
    return int(value) if kind == "integer" else float(value)


def _merged(imports: tuple[Import, ...], attribute: str, own: tuple[Any, ...]) -> tuple[Any, ...]:
    """Imported declarations first, then `own`, as a model file reads them (4.12)."""
    from ..format import _merged as merged

    return merged(imports, own, attribute, "OWL ontology") if imports else own
