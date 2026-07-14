"""Register the Memgraph KV / vector / doc-status backends with LightRAG.

LightRAG resolves storage backends by name through the registry in
``lightrag.kg`` (``STORAGES``, ``STORAGE_IMPLEMENTATIONS`` and
``STORAGE_ENV_REQUIREMENTS``) and rejects any name that is not listed there via
``verify_storage_implementation``. Because LightRAG imports built-in backends
with ``package="lightrag"``, out-of-tree backends must be registered with an
ABSOLUTE module path (``lightrag_memgraph.*``), which LightRAG's
``lazy_external_import`` imports unchanged.

Call :func:`register_memgraph_storage` once before constructing ``LightRAG``.
It is idempotent and safe to call multiple times.
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

_ENV_REQUIREMENTS = ["MEMGRAPH_URI"]

_registered = False


def register_memgraph_storage() -> None:
    """Patch LightRAG's storage registry so it accepts the Memgraph backends by name.

    Idempotent: repeated calls are no-ops after the first successful registration.
    """
    global _registered
    if _registered:
        return

    for name, (storage_type, module_path) in _STORAGES.items():
        # 1) Map the storage name to its (absolute) module path for lazy import.
        _kg.STORAGES[name] = module_path

        # 2) Declare it as a compatible implementation of its storage type so
        #    verify_storage_implementation() accepts the name.
        implementations = _kg.STORAGE_IMPLEMENTATIONS[storage_type]["implementations"]
        if name not in implementations:
            implementations.append(name)

        # 3) Declare its required environment variables.
        _kg.STORAGE_ENV_REQUIREMENTS[name] = list(_ENV_REQUIREMENTS)

    _registered = True
    logger.info("Registered Memgraph KV/vector/doc-status backends with LightRAG")
