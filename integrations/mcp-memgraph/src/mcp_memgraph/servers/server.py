import re
from typing import Any

from fastmcp import Context, FastMCP
from starlette.responses import JSONResponse

from mcp_memgraph.auth import current_session_auth
from mcp_memgraph.config import get_auth_config, get_mcp_config, get_memgraph_config
from mcp_memgraph.tenant_routing import UnknownTenantError, get_registry
from memgraph_toolbox.api.memgraph import Memgraph
from memgraph_toolbox.tools.cypher import CypherTool
from memgraph_toolbox.tools.schema import EnumSchemaTool, NodeSchemaTool, RelationshipSchemaTool, SearchSchemaTool
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


@mcp.tool()
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


@mcp.tool()
def run_cypher_query(query: str, ctx: Context | None = None) -> list[dict[str, Any]]:
    """
    Run a Cypher query on Memgraph. Write operations are blocked if
    server is in read-only mode.

    Args:
        query: The Cypher query to execute on the Memgraph database.
    """
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


@mcp.tool()
def search_schema(pattern: str, ctx: Context | None = None) -> list[dict[str, Any]]:
    """
    Search the entire graph schema (nodes, relationships and enums) by a regex pattern.
    Matches against labels, types, descriptions, and property keys/descriptions.

    Args:
        pattern: A case-insensitive regex pattern to search for (e.g. "person", "pay.*ment").
    """
    logger.info("Searching schema with pattern: %s", pattern)
    return _safe_call(
        lambda: SearchSchemaTool(db=_get_db()).call({"pattern": pattern}),
        on_error="Error searching schema",
    )


@mcp.tool()
def get_node_schema(node_labels: list[str], ctx: Context | None = None) -> list[dict[str, Any]]:
    """
    Get the full schema definition of a node by its labels. Returns properties,
    indexes, constraints, and all relationships where this node appears.

    Args:
        node_labels: The labels of the node to get the details of.
    """
    logger.info("Fetching node schema for labels: %s", node_labels)
    return _safe_call(
        lambda: NodeSchemaTool(db=_get_db()).call({"node_labels": node_labels}),
        on_error="Error fetching node schema",
    )


@mcp.tool()
def get_relationship_schema(
    relationship_type: str,
    start_node_labels: list[str],
    end_node_labels: list[str],
    ctx: Context | None = None,
) -> list[dict[str, Any]]:
    """
    Get the full schema definition of a relationship by its type and connected
    node labels. Returns properties and indexes.

    Args:
        relationship_type: The type of the relationship to get the details of.
        start_node_labels: The labels of the start node of the relationship.
        end_node_labels: The labels of the end node of the relationship.
    """
    logger.info("Fetching relationship schema for type: %s", relationship_type)
    return _safe_call(
        lambda: RelationshipSchemaTool(db=_get_db()).call(
            {
                "relationship_type": relationship_type,
                "start_node_labels": start_node_labels,
                "end_node_labels": end_node_labels,
            }
        ),
        on_error="Error fetching relationship schema",
    )


@mcp.tool()
def get_enum_schema(enum_name: str, ctx: Context | None = None) -> list[dict[str, Any]]:
    """
    Get the schema definition of an enum by its name. Returns the enum name and its values.

    Args:
        enum_name: The name of the enum to get the details of.
    """
    logger.info("Fetching enum schema for: %s", enum_name)
    return _safe_call(
        lambda: EnumSchemaTool(db=_get_db()).call({"enum_name": enum_name}),
        on_error="Error fetching enum schema",
    )
