"""Context Graph's Enterprise edition: personal and shared memory spaces for many users on one Memgraph.

Selected by ``edition = "enterprise"`` in the Context Graph config file, which
makes :class:`EnterpriseSpaces` the space resolver. The admin side provisions
spaces from a spec (:mod:`.spec`, :mod:`.provision`, ``context-graph-spaces
plan|apply``).
"""

from .provision import Applied, Plan, apply, observed_state, plan
from .resolver import EnterpriseSpaces
from .spec import Spec, SpecError, load_spec

__all__ = ["Applied", "EnterpriseSpaces", "Plan", "Spec", "SpecError", "apply", "load_spec", "observed_state", "plan"]
