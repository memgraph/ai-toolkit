"""model_to_owl: a model written as OWL, the import's table backwards (docs/contract.md, section 10).

Read back by OwlImportStrategy, the model is the same, up to order: OWL keeps
none, so the import lists types, properties and individuals in IRI order and
a multi-valued annotation's values sorted.
"""

from dataclasses import fields, replace
from typing import Any

from ..errors import ModelError
from ..format import _own_types, property_to_mapping
from ..types import HygmModel, Individual, NodeType, Property, RelationRestriction, RelationType
from .owl import HYGM, json_literal

#: Where a model's types, properties and individuals get an IRI when they have none.
DEFAULT_BASE = "urn:hygm:model#"

#: What each property type is written as (4.6); the import's table read backwards.
_XSD_TYPES = {
    "string": "string",
    "integer": "integer",
    "float": "double",
    "decimal": "decimal",
    "boolean": "boolean",
    "date": "date",
    "datetime": "dateTime",
}

#: The facets a subclass restriction narrows an inherited property by.
_RESTRICTABLE = {"required", "min_count", "max_count", "value", "values", "minimum", "maximum"}


def model_to_owl(model: HygmModel, *, format: str = "turtle", base: str = DEFAULT_BASE) -> str:
    """`model` as an OWL ontology that OwlImportStrategy reads back as the same model, up to order.

    Everything OWL has a construct for is written as that construct (the
    table in docs/contract.md, section 10); the rest as `hygm:` annotations.
    A type, property or individual without an IRI gets one under `base`,
    recorded as `hygm:base`, so the import knows it was minted and leaves it
    out again. A property is named per type in hygm but once in OWL, so each
    gets its own datatype property, named back by `hygm:name`. A
    redeclaration is written as a subproperty of what it redeclares, or, when
    it keeps that property's IRI, as restrictions on its type. A model with
    imports writes its own types and `owl:imports` the paths, as the model
    file does.

    Args:
        model: The model to write.
        format: Any serialization rdflib writes: turtle, xml, json-ld, ...
        base: The namespace minted IRIs go under; it ends in `#` or `/`.

    Raises:
        ImportError: if rdflib isn't installed.
        ValueError: if `base` doesn't end in `#` or `/`.
        ModelError: `duplicate_label` or `duplicate_individual` if the model
            changes something it imports, as model_to_mapping() refuses it.
    """
    from importlib.util import find_spec

    if find_spec("rdflib") is None:
        raise ImportError("OWL export needs rdflib: install `hygm[owl]`")
    if not base.endswith(("#", "/")):
        raise ValueError(f"base must end in '#' or '/', got {base!r}")
    return _Exporter(model, base).write().serialize(format=format)


def local_name(iri: str) -> str:
    """What follows the last `#`, else the last `/`, of `iri`: the name the import reads it as."""
    return iri.rsplit("#", 1)[-1] if "#" in iri else iri.rstrip("/").rsplit("/", 1)[-1]


class _Exporter:
    def __init__(self, model: HygmModel, base: str) -> None:
        from rdflib import Graph, Namespace
        from rdflib.collection import Collection
        from rdflib.namespace import OWL, RDF, RDFS, SKOS, XSD
        from rdflib.term import BNode, Literal, URIRef

        self.model, self.base = model, base
        self.Collection, self.BNode, self.Literal, self.URIRef = Collection, BNode, Literal, URIRef
        self.OWL, self.RDF, self.RDFS, self.SKOS, self.XSD = OWL, RDF, RDFS, SKOS, XSD
        self.HYGM = Namespace(HYGM)
        self.graph = Graph()
        for prefix, namespace in (("owl", OWL), ("rdfs", RDFS), ("xsd", XSD), ("skos", SKOS), ("hygm", self.HYGM)):
            self.graph.bind(prefix, namespace)
        self.taken = {
            item.iri for item in (*model.node_types, *model.relation_types, *model.individuals) if item.iri is not None
        } | {p.iri for owner in (*model.node_types, *model.relation_types) for p in owner.properties if p.iri}
        self.nodes = {t.label: self.term(t.iri, t.label) for t in model.node_types}
        self.relations = {r.label: self.term(r.iri, r.label) for r in model.relation_types}
        self.properties: dict[tuple[str, str], Any] = {}
        self.restricted: set[tuple[str, str]] = set()
        for node in model.node_types:
            for prop in node.properties:
                self.properties[(node.label, prop.name)] = self.property_term(node, prop)
        self.individuals = {i.id: self.term(i.iri, i.id) for i in model.individuals}
        self.annotations: set[str] = set()

    # IRIs

    def mint(self, name: str) -> str:
        iri, suffix = self.base + name, 1
        while iri in self.taken:
            suffix += 1
            iri = f"{self.base}{name}_{suffix}"
        self.taken.add(iri)
        return iri

    def term(self, iri: str | None, name: str) -> Any:
        return self.URIRef(iri if iri is not None else self.mint(name))

    def property_term(self, node: NodeType, prop: Property) -> Any:
        """A property's IRI: its own, or minted per type. A redeclaration keeping its parent's IRI shares it."""
        inherited = self.model.inherited(node.label, prop.name)
        if prop.iri is not None and inherited is not None and inherited.iri == prop.iri:
            if _restrictable(inherited, prop):
                self.restricted.add((node.label, prop.name))
                return self.URIRef(prop.iri)
            return self.URIRef(self.mint(f"{node.label}_{prop.name}"))
        return self.term(prop.iri, f"{node.label}_{prop.name}")

    def declaring(self, label: str, name: str) -> Any:
        """The IRI of the declaration of `name` nearest `label`: its own, else its nearest ancestor's."""
        for owner in (label, *self.model.ancestors(label)):
            if (owner, name) in self.properties:
                return self.properties[(owner, name)]
        raise ModelError("undeclared_property", f"{label!r} has no property {name!r}")

    # Triples

    def add(self, subject: Any, predicate: Any, value: Any) -> None:
        if str(predicate).startswith(HYGM):
            self.annotations.add(str(predicate))
        self.graph.add((subject, predicate, value))

    def text(self, subject: Any, name: str, item: NodeType | RelationType | Property) -> None:
        """A label, description, names and synonyms; the description even when empty if names stand beside it."""
        if local_name(str(subject)) != name:
            self.add(subject, self.HYGM.name, self.Literal(name))
        if item.description or item.names:
            self.add(subject, self.RDFS.comment, self.Literal(item.description))
        for tag, text in item.names.items():
            self.add(subject, self.RDFS.label, self.Literal(text, lang=tag))
        for synonym in item.synonyms:
            self.add(subject, self.SKOS.altLabel, self.Literal(synonym))
        if item.profiles:
            self.add(subject, self.HYGM.profiles, self.Literal(json_literal(dict(item.profiles))))

    def classes(self, labels: tuple[str, ...]) -> Any:
        """One class, or an `owl:unionOf` of several."""
        if len(labels) == 1:
            return self.nodes[labels[0]]
        union = self.BNode()
        self.add(union, self.RDF.type, self.OWL.Class)
        self.add(union, self.OWL.unionOf, self.list([self.nodes[label] for label in labels]))
        return union

    def list(self, items: list[Any]) -> Any:
        head = self.BNode()
        self.Collection(self.graph, head, items)
        return head

    def restriction(self, owner: Any, prop: Any, predicate: Any, value: Any) -> None:
        restriction = self.BNode()
        self.add(restriction, self.RDF.type, self.OWL.Restriction)
        self.add(restriction, self.OWL.onProperty, prop)
        self.add(restriction, predicate, value)
        self.add(owner, self.RDFS.subClassOf, restriction)

    def count(self, n: int) -> Any:
        return self.Literal(n, datatype=self.XSD.nonNegativeInteger)

    def literal(self, value: Any, kind: str) -> Any:
        if kind == "string":
            return self.Literal(value)
        if kind == "json":
            return self.Literal(json_literal(value), datatype=self.RDF.JSON)
        # Dates are ISO 8601 text in hygm; keep it as written (`Z` stays `Z`) rather than rdflib's canonical form.
        return self.Literal(value, datatype=self.XSD[_XSD_TYPES[kind]], normalize=kind not in ("date", "datetime"))

    # The model

    def write(self) -> Any:
        nodes, relations, individuals = _own_types(self.model)
        header = self.header()
        for node in nodes:
            self.node(node)
        for relation in relations:
            self.relation(relation)
        for individual in individuals:
            self.individual(individual)
        for item in self.model.imports:
            self.add(header, self.OWL.imports, self.URIRef(item.path))
        for annotation in sorted(self.annotations):
            self.graph.add((self.URIRef(annotation), self.RDF.type, self.OWL.AnnotationProperty))
        return self.graph

    def header(self) -> Any:
        ontology = self.model.ontology
        header = self.URIRef(ontology.iri if ontology and ontology.iri else self.base.rstrip("#/"))
        self.add(header, self.RDF.type, self.OWL.Ontology)
        self.add(header, self.HYGM.base, self.Literal(self.base))
        if self.model.profiles:
            self.add(header, self.HYGM.profiles, self.Literal(json_literal(dict(self.model.profiles))))
        if ontology is None:
            return header
        self.add(header, self.HYGM.ontologyBlock, self.Literal(True))
        for key, predicate in (
            ("name", self.RDFS.label),
            ("description", self.RDFS.comment),
            ("version_info", self.OWL.versionInfo),
        ):
            if getattr(ontology, key) is not None:
                self.add(header, predicate, self.Literal(getattr(ontology, key)))
        for key, predicate in (
            ("includes", self.HYGM.includes),
            ("excludes", self.HYGM.excludes),
            ("competency_questions", self.HYGM.competencyQuestion),
            ("maintainers", self.HYGM.maintainer),
        ):
            for value in getattr(ontology, key):
                self.add(header, predicate, self.Literal(value))
        if ontology.conventions is not None:
            conventions = {f.name: getattr(ontology.conventions, f.name) for f in fields(ontology.conventions)}
            self.add(
                header,
                self.HYGM.conventions,
                self.Literal(json_literal({k: v for k, v in conventions.items() if v is not None})),
            )
        return header

    def node(self, node: NodeType) -> None:
        cls = self.nodes[node.label]
        self.add(cls, self.RDF.type, self.OWL.Class)
        self.text(cls, node.label, node)
        for parent in node.parents:
            self.add(cls, self.RDFS.subClassOf, self.nodes[parent])
        if node.abstract:
            self.add(cls, self.HYGM.abstract, self.Literal(True))
        for other in node.disjoint_with:
            self.add(cls, self.OWL.disjointWith, self.nodes[other])
        if node.key:
            self.add(cls, self.OWL.hasKey, self.list([self.declaring(node.label, name) for name in node.key]))
        for restriction in node.relation_restrictions:
            self.relation_restriction(cls, restriction)
        for prop in node.properties:
            if (node.label, prop.name) in self.restricted:
                self.narrowing(node, prop)
            else:
                self.datatype_property(node, prop)

    def relation_restriction(self, cls: Any, restriction: RelationRestriction) -> None:
        relation = self.relations[restriction.relation]
        if restriction.end_labels is not None:
            self.restriction(cls, relation, self.OWL.allValuesFrom, self.classes(restriction.end_labels))
        if restriction.min_count is not None:
            self.restriction(cls, relation, self.OWL.minCardinality, self.count(restriction.min_count))
        if restriction.max_count is not None:
            self.restriction(cls, relation, self.OWL.maxCardinality, self.count(restriction.max_count))

    def datatype_property(self, node: NodeType, prop: Property) -> None:
        cls, term = self.nodes[node.label], self.properties[(node.label, prop.name)]
        self.add(term, self.RDF.type, self.OWL.DatatypeProperty)
        self.add(term, self.RDFS.domain, cls)
        self.text(term, prop.name, prop)
        self.add(term, self.RDFS.range, self.data_range(prop))
        for parent in node.parents:
            if any(p.name == prop.name for p in self.model.properties_of(parent)):
                self.add(term, self.RDFS.subPropertyOf, self.declaring(parent, prop.name))
        if prop.max_count == 1:
            self.add(term, self.RDF.type, self.OWL.FunctionalProperty)
        elif prop.max_count is None:
            self.add(term, self.HYGM.multiValued, self.Literal(True))
        else:
            self.restriction(cls, term, self.OWL.maxCardinality, self.count(prop.max_count))
        if prop.least_count:
            self.restriction(cls, term, self.OWL.minCardinality, self.count(prop.least_count))
        self.values(cls, term, prop)
        if prop.default is not None:
            default = list(prop.default) if isinstance(prop.default, tuple) else prop.default
            self.add(term, self.HYGM.default, self.Literal(json_literal(default)))
        if prop.unit is not None:
            self.add(term, self.HYGM.unit, self.Literal(prop.unit))

    def values(self, cls: Any, term: Any, prop: Property) -> None:
        if prop.value is None:
            return
        for value in prop.value if isinstance(prop.value, tuple) else (prop.value,):
            self.restriction(cls, term, self.OWL.hasValue, self.literal(value, prop.type))

    def data_range(self, prop: Property) -> Any:
        if prop.type == "json":
            return self.RDF.JSON
        datatype = self.XSD[_XSD_TYPES[prop.type]]
        if prop.values is not None:
            return self.one_of(prop.values)
        if prop.minimum is None and prop.maximum is None:
            return datatype
        return self.bounded(datatype, prop)

    def one_of(self, values: tuple[str, ...]) -> Any:
        data_range = self.BNode()
        self.add(data_range, self.RDF.type, self.RDFS.Datatype)
        self.add(data_range, self.OWL.oneOf, self.list([self.Literal(v) for v in values]))
        return data_range

    def bounded(self, datatype: Any, prop: Property) -> Any:
        facets = []
        for bound, facet in ((prop.minimum, self.XSD.minInclusive), (prop.maximum, self.XSD.maxInclusive)):
            if bound is not None:
                node = self.BNode()
                self.add(node, facet, self.literal(bound, prop.type))
                facets.append(node)
        data_range = self.BNode()
        self.add(data_range, self.RDF.type, self.RDFS.Datatype)
        self.add(data_range, self.OWL.onDatatype, datatype)
        self.add(data_range, self.OWL.withRestrictions, self.list(facets))
        return data_range

    def narrowing(self, node: NodeType, prop: Property) -> None:
        """A redeclaration keeping its parent's IRI, as restrictions on its type (the import's reading of them)."""
        cls, term = self.nodes[node.label], self.properties[(node.label, prop.name)]
        inherited = self.model.inherited(node.label, prop.name)
        if inherited is None:  # unreachable: property_term() only marks inherited properties
            return
        if prop.least_count != inherited.least_count:
            self.restriction(cls, term, self.OWL.minCardinality, self.count(prop.least_count))
        if prop.max_count != inherited.max_count and prop.max_count is not None:
            self.restriction(cls, term, self.OWL.maxCardinality, self.count(prop.max_count))
        if prop.value != inherited.value and prop.value is not None and not isinstance(prop.value, tuple):
            self.restriction(cls, term, self.OWL.hasValue, self.literal(prop.value, prop.type))
        if prop.values != inherited.values and prop.values is not None:
            self.restriction(cls, term, self.OWL.allValuesFrom, self.one_of(prop.values))
        if (prop.minimum, prop.maximum) != (inherited.minimum, inherited.maximum):
            datatype = self.XSD[_XSD_TYPES[prop.type]]
            self.restriction(cls, term, self.OWL.allValuesFrom, self.bounded(datatype, prop))

    def relation(self, relation: RelationType) -> None:
        term = self.relations[relation.label]
        self.add(term, self.RDF.type, self.OWL.ObjectProperty)
        self.text(term, relation.label, relation)
        if relation.start_labels:
            self.add(term, self.RDFS.domain, self.classes(relation.start_labels))
        if relation.end_labels:
            self.add(term, self.RDFS.range, self.classes(relation.end_labels))
        if relation.properties:
            properties = [property_to_mapping(p) for p in relation.properties]
            self.add(term, self.HYGM.properties, self.Literal(json_literal(properties)))
        if relation.max_count == 1:
            self.add(term, self.RDF.type, self.OWL.FunctionalProperty)
        elif relation.max_count is not None:
            self.add(term, self.HYGM.maxCount, self.count(relation.max_count))
        if relation.min_count:
            self.add(term, self.HYGM.minCount, self.count(relation.min_count))
        if relation.inverse_of is not None:
            self.add(term, self.OWL.inverseOf, self.relations[relation.inverse_of])
        if relation.symmetric:
            self.add(term, self.RDF.type, self.OWL.SymmetricProperty)

    def individual(self, individual: Individual) -> None:
        term = self.individuals[individual.id]
        self.add(term, self.RDF.type, self.OWL.NamedIndividual)
        self.add(term, self.RDF.type, self.nodes[individual.type])
        if local_name(str(term)) != individual.id:
            self.add(term, self.HYGM.name, self.Literal(individual.id))
        effective = {p.name: p for p in self.model.properties_of(individual.type)}
        for name, value in individual.properties.items():
            prop = self.declaring(individual.type, name)
            declared = effective.get(name)
            kind = declared.type if declared else "string"
            values = list(value) if declared and declared.multi_valued else [value]
            if not values:
                self.add(term, self.HYGM.noValues, self.Literal(name))
            for one in values:
                self.add(term, prop, self.literal(one, kind))
        for relation, targets in individual.relations.items():
            for target in targets:
                self.add(term, self.relations[relation], self.individuals[target])


def _restrictable(inherited: Property, prop: Property) -> bool:
    """Whether `prop` differs from `inherited` only in facets a subclass restriction narrows."""
    narrowed = {name: getattr(prop, name) for name in _RESTRICTABLE}
    return replace(inherited, **narrowed) == prop and not isinstance(prop.value, tuple)
