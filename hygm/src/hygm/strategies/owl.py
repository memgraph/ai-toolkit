"""OwlImportStrategy: a model imported from a standard OWL ontology."""

from pathlib import Path
from typing import Any

from ..identifiers import is_valid_identifier
from ..types import HygmModel, NodeType, RelationType
from ..validation import validate_model


class OwlImportStrategy:
    """Maps an OWL ontology onto a HygmModel, mechanically:

    - each `owl:Class` (and any class a property's domain or range names) is a
      NodeType, labelled by its IRI's local name, described by its
      `rdfs:comment` (else `rdfs:label`);
    - each `owl:ObjectProperty` is a RelationType, its `rdfs:domain` and
      `rdfs:range` the start and end labels. A domain or range may be one class
      or an `owl:unionOf` of several; both become a tuple of labels, and a
      missing one means any declared label.

    Every node type gets the default identity; OWL has no counterpart.
    Needs the `hygm[owl]` extra (rdflib).
    """

    def create_model(self, path: str | Path, format: str | None = None) -> HygmModel:
        """Parse the ontology at `path` (any serialization rdflib reads; `format` overrides its guess).

        Raises:
            ImportError: if rdflib isn't installed.
            ValueError: if the file cannot be parsed, a local name isn't a
                valid Cypher identifier, or the model fails validate_model().
        """
        try:
            from rdflib import Graph
            from rdflib.collection import Collection
            from rdflib.namespace import OWL, RDF, RDFS
            from rdflib.term import BNode, URIRef
        except ImportError as e:
            raise ImportError("OwlImportStrategy needs rdflib: install `hygm[owl]`") from e

        graph = Graph()
        try:
            graph.parse(str(path), format=format)
        except Exception as e:
            raise ValueError(f"Could not parse OWL ontology {path}: {e}") from e

        def local_name(term: Any) -> str:
            name = str(term)
            name = name.rsplit("#", 1)[-1] if "#" in name else name.rstrip("/").rsplit("/", 1)[-1]
            if not is_valid_identifier(name):
                raise ValueError(f"OWL ontology {path}: {term} has local name {name!r}, not a valid Cypher identifier")
            return name

        def text(term: Any) -> str:
            for predicate in (RDFS.comment, RDFS.label):
                value = graph.value(term, predicate)
                if value is not None:
                    return str(value)
            return ""

        def classes_of(term: Any) -> list[Any]:
            if isinstance(term, URIRef):
                return [term]
            if isinstance(term, BNode):
                union = graph.value(term, OWL.unionOf)
                if union is not None:
                    return [member for member in Collection(graph, union) if isinstance(member, URIRef)]
            return []

        def endpoint(prop: Any, predicate: Any) -> list[Any]:
            return [cls for value in graph.objects(prop, predicate) for cls in classes_of(value)]

        properties = sorted(set(graph.subjects(RDF.type, OWL.ObjectProperty)), key=str)
        declared = [c for c in graph.subjects(RDF.type, OWL.Class) if isinstance(c, URIRef)]
        referenced = [c for p in properties for pred in (RDFS.domain, RDFS.range) for c in endpoint(p, pred)]
        classes = sorted(set(declared) | set(referenced), key=str)

        node_types = tuple(NodeType(label=local_name(c), description=text(c)) for c in classes)
        relation_types = tuple(
            RelationType(
                label=local_name(p),
                description=text(p),
                start_labels=tuple(dict.fromkeys(local_name(c) for c in endpoint(p, RDFS.domain))),
                end_labels=tuple(dict.fromkeys(local_name(c) for c in endpoint(p, RDFS.range))),
            )
            for p in properties
        )
        model = HygmModel(node_types=node_types, relation_types=relation_types)
        result = validate_model(model)
        if not result.success:
            problems = "; ".join(issue.message for issue in result.critical_issues)
            raise ValueError(f"OWL ontology {path} is not a valid model: {problems}")
        return model
