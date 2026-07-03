import re
from typing import Any

from fastmcp import Context, FastMCP
from starlette.responses import JSONResponse

from mcp_memgraph.auth import current_session_auth
from mcp_memgraph.config import get_auth_config, get_mcp_config, get_memgraph_config
from mcp_memgraph.tenant_routing import UnknownTenantError, get_registry
from memgraph_toolbox.api.memgraph import Memgraph
from memgraph_toolbox.tools.betweenness_centrality import (
    BetweennessCentralityTool,
)
from memgraph_toolbox.tools.config import ShowConfigTool
from memgraph_toolbox.tools.constraint import ShowConstraintInfoTool
from memgraph_toolbox.tools.cypher import CypherTool
from memgraph_toolbox.tools.index import ShowIndexInfoTool
from memgraph_toolbox.tools.node_neighborhood import NodeNeighborhoodTool
from memgraph_toolbox.tools.node_vector_search import NodeVectorSearchTool
from memgraph_toolbox.tools.page_rank import PageRankTool
from memgraph_toolbox.tools.procedures import ShowProceduresTool
from memgraph_toolbox.tools.schema import ShowSchemaInfoTool
from memgraph_toolbox.tools.storage import ShowStorageInfoTool
from memgraph_toolbox.tools.trigger import ShowTriggersTool
from memgraph_toolbox.utils.logger import logger_init

# Get configuration instances
memgraph_config = get_memgraph_config()
mcp_config = get_mcp_config()
auth_config = get_auth_config()

# Configure logging
logger = logger_init("mcp-memgraph")

# Initialize FastMCP server
mcp = FastMCP("mcp-memgraph")

# Read-only mode flag (from config)
READ_ONLY_MODE = mcp_config.read_only

# Patterns for write operations in Cypher
WRITE_PATTERNS = [
    r"\bCREATE\b",
    r"\bMERGE\b",
    r"\bDELETE\b",
    r"\bREMOVE\b",
    r"\bSET\b",
    r"\bDROP\b",
    r"\bCREATE\s+INDEX\b",
    r"\bDROP\s+INDEX\b",
    r"\bCREATE\s+CONSTRAINT\b",
    r"\bDROP\s+CONSTRAINT\b",
]


def is_write_query(query: str) -> bool:
    """Check if a Cypher query contains write operations"""
    query_upper = query.upper()
    return any(re.search(pattern, query_upper) for pattern in WRITE_PATTERNS)


# Per-tenant client registry. Falls back to a single default client when auth
# is disabled or when an unauthenticated caller (stdio / test import) reaches a
# tool function directly.
_registry = get_registry(memgraph_config, auth_config)

if auth_config.enabled:
    logger.info(
        "Multi-tenant mode: catalog=%s, single Memgraph backend at %s",
        sorted(_registry.catalog),
        memgraph_config.url,
    )
else:
    logger.info(
        "Single-DB mode: connecting to Memgraph db '%s' at %s with user '%s'",
        memgraph_config.database,
        memgraph_config.url,
        memgraph_config.username,
    )
logger.info("Read-only mode: %s", READ_ONLY_MODE)


def _get_db() -> Memgraph:
    """Resolve the Memgraph client for the current request.

    When auth is enabled and an authenticated SessionAuth is present on the
    request (set by AuthMiddleware via a contextvar), route to that caller's
    currently-active tenant. Otherwise — auth disabled, or stdio / direct
    test invocation — return the legacy single-DB client.
    """
    if auth_config.enabled:
        sa = current_session_auth()
        if sa is not None:
            return _registry.get_for(sa.current_tenant)
    return _registry.get_default()


def _tenant_unavailable_error(name: str) -> dict[str, Any]:
    """Uniform error for 'name not in your token' AND 'name not in catalog'.

    Returning the same shape for both prevents enumeration of the server-side
    catalog from a caller's perspective.
    """
    sa = current_session_auth()
    allowed = sorted(sa.allowed_tenants) if sa is not None else []
    return {"error": "database not available", "requested": name, "allowed": allowed}


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request):
    return JSONResponse({"status": "healthy", "service": "mcp-server"})


# ---------------------------------------------------------------------------
# Tenant-management tools (only meaningful when MCP_AUTH_ENABLED=true)
# ---------------------------------------------------------------------------


@mcp.tool(annotations={"readOnlyHint": True})
def list_databases(ctx: Context | None = None) -> list[dict[str, Any]]:
    """List databases this session can access. The active one is flagged."""
    sa = current_session_auth()
    if sa is None:
        # Auth disabled or stdio: there's exactly one database.
        return [{"name": memgraph_config.database, "current": True}]
    return [{"name": name, "current": (name == sa.current_tenant)} for name in sorted(sa.allowed_tenants)]


@mcp.tool()
def use_database(name: str, ctx: Context | None = None) -> dict[str, Any]:
    """Switch the active database for this MCP session.

    The new name must be one of the databases your token authorizes. Switching
    persists for the duration of the MCP session and does not require a new
    login.
    """
    sa = current_session_auth()
    if sa is None:
        return {
            "error": "auth disabled — there is only one database",
            "current": memgraph_config.database,
        }
    if name not in sa.allowed_tenants:
        return _tenant_unavailable_error(name)
    sa.current_tenant = name
    return {"status": "ok", "current": name}


# ---------------------------------------------------------------------------
# Existing data tools (now routed per-tenant when auth is enabled)
# ---------------------------------------------------------------------------


def _safe_call(fn, *, on_error: str):
    """Run *fn*, translating UnknownTenantError into the uniform shape."""
    try:
        return fn()
    except UnknownTenantError as e:
        return [_tenant_unavailable_error(str(e))]
    except Exception as e:
        return [{"error": f"{on_error}: {e!s}"}]


@mcp.tool(annotations={"readOnlyHint": READ_ONLY_MODE})
def run_query(query: str, ctx: Context | None = None) -> list[dict[str, Any]]:
    """Run a Cypher query on Memgraph. Write operations are blocked if
    server is in read-only mode."""
    logger.info("Running query: %s", query)

    # Check if query is a write operation in read-only mode
    if READ_ONLY_MODE and is_write_query(query):
        logger.warning("Write operation blocked in read-only mode: %s", query)
        return [
            {
                "error": "Write operations are not allowed in read-only mode",
                "query": query,
                "mode": "read-only",
                "hint": "Set MCP_READ_ONLY=false to enable write operations",
            }
        ]

    return _safe_call(lambda: CypherTool(db=_get_db()).call({"query": query}), on_error="Error running query")


@mcp.tool(annotations={"readOnlyHint": True})
def get_configuration(ctx: Context | None = None) -> list[dict[str, Any]]:
    """Get Memgraph configuration information"""
    logger.info("Fetching Memgraph configuration...")
    return _safe_call(lambda: ShowConfigTool(db=_get_db()).call({}), on_error="Error fetching configuration")


@mcp.tool(annotations={"readOnlyHint": True})
def get_index(ctx: Context | None = None) -> list[dict[str, Any]]:
    """Get Memgraph index information"""
    logger.info("Fetching Memgraph index...")
    return _safe_call(lambda: ShowIndexInfoTool(db=_get_db()).call({}), on_error="Error fetching index")


@mcp.tool(annotations={"readOnlyHint": True})
def get_constraint(ctx: Context | None = None) -> list[dict[str, Any]]:
    """Get Memgraph constraint information"""
    logger.info("Fetching Memgraph constraint...")
    return _safe_call(lambda: ShowConstraintInfoTool(db=_get_db()).call({}), on_error="Error fetching constraint")


@mcp.tool(annotations={"readOnlyHint": True})
def get_schema(ctx: Context | None = None) -> list[dict[str, Any]]:
    """Get Memgraph schema information"""
    logger.info("Fetching Memgraph schema...")
    return _safe_call(lambda: ShowSchemaInfoTool(db=_get_db()).call({}), on_error="Error fetching schema")


@mcp.tool(annotations={"readOnlyHint": True})
def get_storage(ctx: Context | None = None) -> list[dict[str, Any]]:
    """Get Memgraph storage information"""
    logger.info("Fetching Memgraph storage...")
    return _safe_call(lambda: ShowStorageInfoTool(db=_get_db()).call({}), on_error="Error fetching storage")


@mcp.tool(annotations={"readOnlyHint": True})
def get_triggers(ctx: Context | None = None) -> list[dict[str, Any]]:
    """Get Memgraph triggers information"""
    logger.info("Fetching Memgraph triggers...")
    return _safe_call(lambda: ShowTriggersTool(db=_get_db()).call({}), on_error="Error fetching triggers")


@mcp.tool(annotations={"readOnlyHint": True})
def get_procedures(ctx: Context | None = None) -> list[dict[str, Any]]:
    """List all available Memgraph procedures (query modules).

    Returns information about all available procedures including MAGE algorithms
    and custom query modules. Each procedure includes its name, signature, and
    whether it performs write operations. Use this to discover available graph
    algorithms and utility functions before executing them."""
    logger.info("Fetching Memgraph procedures...")
    return _safe_call(lambda: ShowProceduresTool(db=_get_db()).call({}), on_error="Error fetching procedures")


@mcp.tool(annotations={"readOnlyHint": True})
def get_betweenness_centrality(ctx: Context | None = None) -> list[dict[str, Any]]:
    """Get betweenness centrality information"""
    logger.info("Fetching betweenness centrality...")
    return _safe_call(
        lambda: BetweennessCentralityTool(db=_get_db()).call({}),
        on_error="Error fetching betweenness centrality",
    )


@mcp.tool(annotations={"readOnlyHint": True})
def get_page_rank(ctx: Context | None = None) -> list[dict[str, Any]]:
    """Get page rank information"""
    logger.info("Fetching page rank...")
    return _safe_call(lambda: PageRankTool(db=_get_db()).call({}), on_error="Error fetching page rank")


@mcp.tool(annotations={"readOnlyHint": True})
def get_node_neighborhood(
    node_id: str,
    max_distance: int = 1,
    limit: int = 100,
    ctx: Context | None = None,
) -> list[dict[str, Any]]:
    """Find nodes within a specified distance from a given node"""
    logger.info(
        "Finding neighborhood for node %s with max distance %s",
        node_id,
        max_distance,
    )
    return _safe_call(
        lambda: NodeNeighborhoodTool(db=_get_db()).call(
            {"node_id": node_id, "max_distance": max_distance, "limit": limit}
        ),
        on_error="Error finding node neighborhood",
    )


@mcp.tool(annotations={"readOnlyHint": True})
def search_node_vectors(
    index_name: str,
    query_vector: list[float],
    limit: int = 10,
    ctx: Context | None = None,
) -> list[dict[str, Any]]:
    """Perform vector similarity search on nodes in Memgraph"""
    logger.info("Performing vector search on index %s with limit %s", index_name, limit)
    return _safe_call(
        lambda: NodeVectorSearchTool(db=_get_db()).call(
            {
                "index_name": index_name,
                "query_vector": query_vector,
                "limit": limit,
            }
        ),
        on_error="Error performing vector search",
    )
