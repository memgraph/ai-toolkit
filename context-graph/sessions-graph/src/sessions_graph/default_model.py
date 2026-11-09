"""The conversational default model a user's graph is extracted under, and every learned model starts from (#436)."""

from dataclasses import replace
from importlib.resources import as_file, files

from hygm import HygmModel, ManualStrategy

from .learning import CORE_LABELS, VERSION_1


def default_model() -> HygmModel:
    """The generic default: the fixed core, Organization/Location/Event, the catch-alls, and generic relations."""
    with as_file(files("sessions_graph") / "default_ontology.yaml") as path:
        return ManualStrategy(version_1_profiles=VERSION_1).create_model(path)


def with_core(model: HygmModel) -> HygmModel:
    """`model` with any core node type it lacks added from the default, after its own types.

    A core type `model` already declares is kept as `model` declares it.
    """
    declared = set(model.node_labels())
    default = default_model()
    missing = tuple(t for t in default.node_types if t.label in CORE_LABELS and t.label not in declared)
    # The core brings the default's conversation roles with it, unless `model` names its own.
    return replace(model, node_types=model.node_types + missing, profiles={**default.profiles, **model.profiles})
