"""Register the Memgraph KV / vector / doc-status backends with LightRAG.

LightRAG only accepts storage backends listed in its ``lightrag.kg`` registry,
and imports out-of-tree backends by an ABSOLUTE module path (built-ins load
with ``package="lightrag"``, so a relative path wouldn't resolve).

Call :func:`register_memgraph_storage` once before constructing ``LightRAG``;
it's idempotent.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

import lightrag.kg as _kg
from lightrag.utils import logger

# storage name -> (storage type, absolute module path)
_STORAGES = {
    "MemgraphKVStorage": ("KV_STORAGE", "lightrag_memgraph.kv_impl"),
    "MemgraphVectorStorage": ("VECTOR_STORAGE", "lightrag_memgraph.vector_impl"),
    "MemgraphDocStatusStorage": ("DOC_STATUS_STORAGE", "lightrag_memgraph.docstatus_impl"),
}

# Canonical toolbox connection env var; storages resolve it via memgraph_env.
_ENV_REQUIREMENTS = ["MEMGRAPH_URL"]

_registered = False


class LightRAGCompatibilityError(RuntimeError):
    """lightrag.kg's internal registry doesn't have the shape this integration expects.

    STORAGES, STORAGE_IMPLEMENTATIONS and STORAGE_ENV_REQUIREMENTS are undocumented
    lightrag.kg internals, not a public plugin API, so a lightrag-hku release can
    change their shape without notice (pyproject.toml pins lightrag-hku<1.6 for this
    reason). Raised instead of letting a raw KeyError/AttributeError surface from
    deep inside registration.
    """


def _installed_lightrag_version() -> str:
    try:
        return version("lightrag-hku")
    except PackageNotFoundError:
        return "unknown"


def _check_lightrag_kg_shape() -> None:
    problem = None
    if not hasattr(_kg, "STORAGES"):
        problem = "lightrag.kg.STORAGES no longer exists"
    elif not hasattr(_kg, "STORAGE_IMPLEMENTATIONS"):
        problem = "lightrag.kg.STORAGE_IMPLEMENTATIONS no longer exists"
    elif not hasattr(_kg, "STORAGE_ENV_REQUIREMENTS"):
        problem = "lightrag.kg.STORAGE_ENV_REQUIREMENTS no longer exists"
    else:
        for storage_type in {storage_type for storage_type, _ in _STORAGES.values()}:
            entry = _kg.STORAGE_IMPLEMENTATIONS.get(storage_type)
            if not isinstance(entry, dict) or "implementations" not in entry:
                problem = (
                    f"lightrag.kg.STORAGE_IMPLEMENTATIONS[{storage_type!r}] is missing or has no 'implementations' list"
                )
                break

    if problem:
        raise LightRAGCompatibilityError(
            f"lightrag-memgraph is incompatible with the installed lightrag-hku "
            f"{_installed_lightrag_version()}: {problem}. This integration is "
            "pinned to lightrag-hku>=1.5,<1.6 in pyproject.toml precisely because "
            "it monkey-patches these undocumented lightrag.kg internals; install "
            "a version within that range, or update "
            "lightrag_memgraph.registry.register_memgraph_storage() for the new "
            "lightrag.kg layout."
        )


def register_memgraph_storage() -> None:
    """Patch LightRAG's storage registry to accept the Memgraph backends by name. Idempotent."""
    global _registered
    if _registered:
        return

    _check_lightrag_kg_shape()

    for name, (storage_type, module_path) in _STORAGES.items():
        _kg.STORAGES[name] = module_path  # absolute module path for lazy import

        implementations = _kg.STORAGE_IMPLEMENTATIONS[storage_type]["implementations"]
        if name not in implementations:
            implementations.append(name)  # accepted by verify_storage_implementation()

        _kg.STORAGE_ENV_REQUIREMENTS[name] = list(_ENV_REQUIREMENTS)

    _registered = True
    logger.info("Registered Memgraph KV/vector/doc-status backends with LightRAG")
