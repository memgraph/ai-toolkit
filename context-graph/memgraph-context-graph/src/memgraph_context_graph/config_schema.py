"""The config file's keys: one table that parsing, rendering and ``config get/set/show`` all read.

Each :class:`ConfigKey` names one ``section.key``, its type, whether it is a
secret, what leaving it unset means, and where :class:`HookConfig` keeps it:
a typed field, or the section's settings dict (``[capture]``, ``[recall]``,
``[extraction]``), which the component that owns those keys validates when it
reads them. Adding a key is one entry here; ``[redact]`` is the one free-form
section, since its keys are pattern names the user chooses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from memgraph_context_graph.capture_policy import DEFAULT_MAX_CHARS
from memgraph_context_graph.connectors import DEFAULT_CONNECTORS
from memgraph_context_graph.spaces import EDITION_COMMUNITY, EDITION_ENTERPRISE, EDITIONS

if TYPE_CHECKING:
    from collections.abc import Callable

Kind = Literal["str", "bool", "list", "int", "float"]

#: The boolean spellings parse_bool_flag accepts, shared with ``config set`` validation.
TRUTHY_VALUES = frozenset({"1", "true", "yes", "on"})
FALSY_VALUES = frozenset({"0", "false", "no", "off"})

#: ``[memory] backend`` values: the graph, or the harness's built-in memory.
MEMORY_BACKEND_GRAPH = "context-graph"
MEMORY_BACKEND_NATIVE = "native"
MEMORY_BACKENDS = frozenset({MEMORY_BACKEND_GRAPH, MEMORY_BACKEND_NATIVE})

#: Sections whose keys :class:`HookConfig` keeps as written, in the settings dict this names.
SETTINGS_SECTIONS = {
    "capture": "capture_settings",
    "recall": "recall_settings",
    "extraction": "extraction_settings",
}


def parse_list(value: str) -> tuple[str, ...]:
    """A comma-separated value (``"a, b"``) as its non-blank items; the parser has no arrays."""
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _connectors(value: str) -> str:
    from memgraph_context_graph.connectors import supported_connectors

    return ", ".join(supported_connectors(value.split(",")))


@dataclass(frozen=True)
class ConfigKey:
    """One key of the config file."""

    #: ``section.key``, or a bare ``key`` for one above the first section.
    name: str
    #: One line for the rendered file, saying what the key does.
    doc: str
    kind: Kind = "str"
    #: The :class:`HookConfig` attribute holding it; None keeps it in the section's settings dict.
    field: str | None = None
    #: Masked by ``config show/get/set`` and prompted for, not echoed, when ``config set`` gets no value.
    secret: bool = False
    #: What an unset key means, as ``config show`` words it (``unset (<this>)``).
    unset: str = ""
    #: The value written, commented out, for an unset key: its default when it has a literal one.
    example: str = ""
    #: The only values allowed, when the key is an enumeration.
    choices: tuple[str, ...] = ()
    #: Always written, even empty: the keys a user is expected to fill in.
    required: bool = False
    #: Left out of an Enterprise file, which takes the user and their database from the login.
    community_only: bool = False
    #: Normalizes a value beyond its kind's own check; raises ValueError naming what is wrong.
    validate: Callable[[str], str] | None = None

    @property
    def section(self) -> str:
        """The ``[section]`` it lives in; ``""`` for a top-level key."""
        return self.name.rpartition(".")[0]

    @property
    def key(self) -> str:
        """The name inside its section."""
        return self.name.rpartition(".")[2]

    def applies_to(self, edition: str | None) -> bool:
        """Whether a file for *edition* holds this key."""
        return not (self.community_only and edition == EDITION_ENTERPRISE)

    def normalize(self, raw: str) -> str:
        """The value as the file stores it, from what a user typed; ``""`` clears an optional key.

        Raises:
            ValueError: *raw* is not a valid value; the message says what was expected.
        """
        value = raw.strip()
        if not value:
            return ""
        if self.choices and value not in self.choices:
            raise ValueError(f"expected {'/'.join(self.choices)}")
        if self.kind == "bool":
            lowered = value.lower()
            if lowered not in TRUTHY_VALUES | FALSY_VALUES:
                raise ValueError("expected true/false")
            value = "true" if lowered in TRUTHY_VALUES else "false"
        elif self.kind == "int":
            if not value.isdigit():
                raise ValueError("expected a non-negative integer")
            value = str(int(value))
        elif self.kind == "float":
            try:
                float(value)
            except ValueError:
                raise ValueError("expected a number") from None
        elif self.kind == "list":
            value = ", ".join(parse_list(value))
        return self.validate(value) if self.validate is not None else value

    def parse(self, value: str) -> str | bool | tuple[str, ...] | None:
        """The typed :class:`HookConfig` field value for a stored string; ``""`` is unset, except for a list."""
        if self.kind == "list":
            return parse_list(value)
        if not value:
            return None
        if self.kind == "bool":
            return value.strip().lower() in TRUTHY_VALUES
        return value

    def format(self, value: str) -> str:
        """*value* as the right-hand side of a line in the file."""
        if self.kind == "bool" and self.field is not None:
            return value
        if self.section == "redact":
            # Single quotes: a pattern may hold a double quote, and the parser does no escape processing.
            return f"'{value}'"
        return f'"{value}"'


#: Every key, in the order the file and ``config show`` list them.
CONFIG_KEYS: tuple[ConfigKey, ...] = (
    ConfigKey(
        "edition",
        "community (one user, the default) or enterprise (spaces on a licensed Memgraph).",
        field="edition",
        choices=tuple(sorted(EDITIONS)),
        unset=f"defaults to {EDITION_COMMUNITY}",
        example=EDITION_COMMUNITY,
    ),
    ConfigKey(
        "identity.user_id",
        "Who your sessions and memories are filed under.",
        field="user_id",
        required=True,
        community_only=True,
    ),
    ConfigKey("memgraph.url", "The Memgraph your hooks write to.", field="memgraph_url", required=True),
    ConfigKey("memgraph.user", "Memgraph login; empty for none.", field="memgraph_user", required=True),
    ConfigKey("memgraph.password", "Memgraph password.", field="memgraph_password", secret=True, required=True),
    ConfigKey(
        "memgraph.database",
        "The database inside that Memgraph.",
        field="memgraph_database",
        required=True,
        community_only=True,
    ),
    ConfigKey(
        "llm.openai_api_key",
        "Optional: lets reconciliation learn your own ontology (sessions-graph derive).",
        field="openai_api_key",
        secret=True,
        required=True,
    ),
    ConfigKey(
        "llm.anthropic_api_key",
        "Optional, as openai_api_key; Anthropic is preferred when both are set.",
        field="anthropic_api_key",
        secret=True,
        required=True,
    ),
    ConfigKey(
        "llm.derive_model",
        "The model ontology derivation calls.",
        field="derive_model",
        unset="defaults to the provider's model",
    ),
    ConfigKey(
        "capture.connectors",
        "Graph connectors a hook enables when its command names none; empty enables none.",
        kind="list",
        field="connectors",
        unset=f"defaults to {', '.join(DEFAULT_CONNECTORS)}",
        example=", ".join(DEFAULT_CONNECTORS),
        validate=_connectors,
    ),
    ConfigKey(
        "capture.redact",
        "Redact secrets and the [redact] patterns before anything is stored.",
        kind="bool",
        unset="defaults to true",
        example="true",
    ),
    ConfigKey(
        "capture.max_chars",
        "Truncate each captured text to this many characters; 0 for no limit.",
        kind="int",
        unset=f"defaults to {DEFAULT_MAX_CHARS}",
        example=str(DEFAULT_MAX_CHARS),
    ),
    ConfigKey(
        "capture.exclude_tools",
        "Tool names (globs) never captured, e.g. TodoWrite, mcp__vault__*.",
        kind="list",
        unset="none excluded",
    ),
    ConfigKey(
        "capture.exclude_projects",
        "Project directories (globs) never captured, e.g. ~/clients/*.",
        kind="list",
        unset="none excluded",
    ),
    ConfigKey(
        "reconcile.auto_reconcile",
        "Extract entities from each session when it ends; needs no LLM.",
        kind="bool",
        field="auto_reconcile",
        unset="defaults to false",
        example="false",
    ),
    ConfigKey(
        "recall.embedding_model",
        "The model recall embeds with.",
        field="embedding_model",
        unset="defaults to sessions-graph's model",
        example="BAAI/bge-small-en-v1.5",
    ),
    ConfigKey(
        "recall.lanes",
        "Which recall lanes run.",
        kind="list",
        unset="defaults to all",
        example="memories, turns, text, entities, facts, user_facts",
    ),
    ConfigKey("recall.memories_k", "Memory files shown at most.", kind="int", unset="defaults to 5", example="5"),
    ConfigKey("recall.turns_k", "Turns found by vector search.", kind="int", unset="defaults to 8", example="8"),
    ConfigKey("recall.text_k", "Turns found by full-text search.", kind="int", unset="defaults to 8", example="8"),
    ConfigKey("recall.entities_k", "Entities matched.", kind="int", unset="defaults to 15", example="15"),
    ConfigKey("recall.edges_per_entity", "Edges shown per entity.", kind="int", unset="defaults to 6", example="6"),
    ConfigKey("recall.facts_k", "Facts matched.", kind="int", unset="defaults to 15", example="15"),
    ConfigKey("recall.fact_turns_k", "Turns the facts lead to.", kind="int", unset="defaults to 8", example="8"),
    ConfigKey(
        "recall.user_fact_types",
        "Relation types the user_facts lane picks.",
        kind="int",
        unset="defaults to 2",
        example="2",
    ),
    ConfigKey(
        "recall.user_facts_k",
        "Facts the user_facts lane keeps across them.",
        kind="int",
        unset="defaults to 30",
        example="30",
    ),
    ConfigKey(
        "recall.turn_chars",
        "Characters of each turn shown.",
        kind="int",
        unset="defaults to 1500",
        example="1500",
    ),
    ConfigKey(
        "recall.query_instruction",
        "Prepended to the question before embedding it.",
        unset="defaults to the embedding model's own",
    ),
    ConfigKey(
        "ontology.path",
        "A schema file your sessions are extracted under.",
        field="ontology_path",
        unset="the default ontology",
    ),
    ConfigKey(
        "ontology.derive",
        "extend: derivation adds types beside the schema's; off: never derive.",
        field="ontology_derive",
        choices=("extend", "off"),
        unset="defaults to extend",
        example="extend",
    ),
    ConfigKey("extraction.model", "A GLiNER2 checkpoint.", unset="GLiNER2's default"),
    ConfigKey(
        "extraction.entity_threshold",
        "Drop entities below this confidence, 0 to 1.",
        kind="float",
        unset="GLiNER2's default",
    ),
    ConfigKey(
        "extraction.relation_threshold",
        "Drop relations below this confidence, 0 to 1.",
        kind="float",
        unset="GLiNER2's default",
    ),
    ConfigKey("extraction.chunk_size", "Words per window the model reads.", kind="int", unset="GLiNER2's default"),
    ConfigKey("extraction.chunk_overlap", "Words consecutive windows share.", kind="int", unset="GLiNER2's default"),
    ConfigKey(
        "github.token",
        "Overrides the gh login resources-graph's sweep fetches with.",
        field="github_token",
        secret=True,
        unset="uses the gh login",
    ),
    ConfigKey(
        "memory.backend",
        "context-graph makes the graph your harness memory; native leaves the harness's own.",
        field="memory_backend",
        choices=tuple(sorted(MEMORY_BACKENDS)),
        unset=MEMORY_BACKEND_NATIVE,
        example=MEMORY_BACKEND_NATIVE,
    ),
    ConfigKey(
        "session_start.resume",
        "Start each session with a brief of the previous ones in its project.",
        kind="bool",
        field="session_resume",
        unset="defaults to true",
        example="true",
    ),
)

#: Sections in file order; ``redact`` holds no table keys, only the user's own pattern names.
SECTIONS: tuple[str, ...] = (
    "identity",
    "memgraph",
    "llm",
    "capture",
    "redact",
    "reconcile",
    "recall",
    "ontology",
    "extraction",
    "github",
    "memory",
    "session_start",
)

_BY_NAME = {key.name: key for key in CONFIG_KEYS}


def config_key(name: str) -> ConfigKey | None:
    """The key called *name* (``section.key``), or None for an unknown one."""
    return _BY_NAME.get(name)
