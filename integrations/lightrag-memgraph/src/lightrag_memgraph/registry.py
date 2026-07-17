"""Register the Memgraph KV / vector / doc-status backends with LightRAG.

LightRAG only accepts storage backends listed in its ``lightrag.kg`` registry,
and imports out-of-tree backends by an ABSOLUTE module path (built-ins load
with ``package="lightrag"``, so a relative path wouldn't resolve).

Call :func:`register_memgraph_storage` once before constructing ``LightRAG``;
it's idempotent.
"""

from __future__ import annotations

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


def register_memgraph_storage() -> None:
    """Patch LightRAG's storage registry to accept the Memgraph backends by name. Idempotent."""
    global _registered
    if _registered:
        return

    for name, (storage_type, module_path) in _STORAGES.items():
        _kg.STORAGES[name] = module_path  # absolute module path for lazy import

        implementations = _kg.STORAGE_IMPLEMENTATIONS[storage_type]["implementations"]
        if name not in implementations:
            implementations.append(name)  # accepted by verify_storage_implementation()

        _kg.STORAGE_ENV_REQUIREMENTS[name] = list(_ENV_REQUIREMENTS)

    _registered = True
    logger.info("Registered Memgraph KV/vector/doc-status backends with LightRAG")
