"""The structural graph model every strategy produces."""

from dataclasses import dataclass
from typing import Literal

from .identifiers import require_valid_identifier

Identity = Literal["global", "chunk", "span"]

#: How a node type's mentions become nodes (#346, #361):
#:   global -- one node per normalized text across every chunk (named things that recur)
#:   chunk  -- one node per normalized text per chunk (generic nouns)
#:   span   -- one node per mention (values: every "3" in a session is a different fact)
IDENTITIES: tuple[Identity, ...] = ("global", "chunk", "span")


@dataclass(frozen=True)
class NodeType:
    """A node label the model declares.

    Attributes:
        label: The Memgraph label, a valid Cypher identifier.
        description: What the type covers. Extractors that read descriptions
            (GLiNER2's entity specs, LightRAG's prompt) steer on it.
        identity: How mentions of this type are merged into nodes; see IDENTITIES.
    """

    label: str
    description: str = ""
    identity: Identity = "chunk"

    def __post_init__(self) -> None:
        require_valid_identifier(self.label, "node label")
        if self.identity not in IDENTITIES:
            raise ValueError(f"Node type {self.label!r}: identity must be one of {IDENTITIES}, got {self.identity!r}")


@dataclass(frozen=True)
class RelationType:
    """A relationship type and the node labels it may connect.

    Attributes:
        label: The Memgraph relationship type, a valid Cypher identifier.
        description: Documentation only for GLiNER2, whose joint compiler drops
            relation descriptions; the label is its only steering surface (#360).
        start_labels: Labels the relationship may start at. Empty means any
            declared label -- OWL's rdfs:domain, sql2graph's start_node_labels.
        end_labels: Labels the relationship may end at. Empty means any
            declared label -- OWL's rdfs:range.
    """

    label: str
    description: str = ""
    start_labels: tuple[str, ...] = ()
    end_labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_valid_identifier(self.label, "relation type")
        for side in ("start_labels", "end_labels"):
            labels = getattr(self, side)
            if isinstance(labels, str):
                raise TypeError(f"Relation type {self.label!r}: {side} must be a tuple of labels, not a string")
            object.__setattr__(self, side, tuple(labels))

    @property
    def constrained(self) -> bool:
        """Whether either endpoint is restricted beyond "any declared label"."""
        return bool(self.start_labels or self.end_labels)


@dataclass(frozen=True)
class HygmModel:
    """A graph model: node types and the relation types between them."""

    node_types: tuple[NodeType, ...]
    relation_types: tuple[RelationType, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_types", tuple(self.node_types))
        object.__setattr__(self, "relation_types", tuple(self.relation_types))

    def node_labels(self) -> tuple[str, ...]:
        return tuple(t.label for t in self.node_types)

    def relation_labels(self) -> tuple[str, ...]:
        return tuple(t.label for t in self.relation_types)

    def node_type(self, label: str) -> NodeType | None:
        return next((t for t in self.node_types if t.label == label), None)

    def endpoint_labels(self, relation: RelationType, side: Literal["start", "end"]) -> tuple[str, ...]:
        """The labels `relation` may connect at `side`, with "unconstrained" resolved to every declared label.

        Extractors that cannot express an unconstrained endpoint (GLiNER2's
        JointSchema rejects an empty head/tail, #345) consume this rather than
        the raw field.
        """
        labels = relation.start_labels if side == "start" else relation.end_labels
        return labels or self.node_labels()
