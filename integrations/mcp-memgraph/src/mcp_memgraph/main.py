import os
from typing import Literal, cast

from mcp_memgraph.config import get_auth_config, get_mcp_config
from mcp_memgraph.servers import get_server, get_server_info


def main():
    # Get MCP server configuration
    config = get_mcp_config()
    auth_config = get_auth_config()
    auth_config.validate()

    # Determine which server to load (default: server)
    server_name = os.getenv("MCP_SERVER", "server").lower()

    # Validate and load server
    try:
        server_module = get_server(server_name)
        mcp = server_module.mcp
        logger = server_module.logger

        # Get server metadata for logging
        server_info = get_server_info(server_name)
        emoji = server_info["emoji"]
        description = server_info["description"]
        startup_msg = f"{emoji} Starting {description}".strip()
        logger.info(startup_msg)

    except (ValueError, ImportError, AttributeError) as e:
        # Fallback to server on any error
        print(f"Warning: {e}")
        print("Falling back to default server.")

        server_module = get_server("server")
        mcp = server_module.mcp
        logger = server_module.logger

        logger.warning("Failed to load server '%s', using default server", server_name)

    logger.info("Server on %s with transport: %s", config.host, config.transport)

    # Run server with configuration
    # Note: host parameter is only used for HTTP/SSE transports, not stdio
    transport = cast("Literal['stdio', 'streamable-http']", config.transport)

    if config.transport == "stdio":
        if auth_config.enabled:
            logger.warning(
                "MCP_AUTH_ENABLED=true is ignored for stdio transport "
                "(authentication only applies to streamable-http)."
            )
        mcp.run(transport=transport)
        return

    # HTTP transport. If auth is disabled, preserve the original code path
    # exactly so existing dev workflows are unaffected.
    if not auth_config.enabled:
        mcp.run(host=config.host, transport=transport)
        return

    # Auth-enabled HTTP path: wrap FastMCP's Starlette app with the JWT
    # middleware and serve it via uvicorn.
    import uvicorn

    from mcp_memgraph.middleware import AuthMiddleware

    inner_app = _get_streamable_http_app(mcp)
    wrapped = AuthMiddleware(inner_app, auth_config)

    logger.info(
        "🔐 JWT auth enabled: issuer=%s, audience=%s, catalog=%s",
        auth_config.issuer,
        auth_config.audience,
        sorted(auth_config.tenant_catalog),
    )
    uvicorn.run(wrapped, host=config.host, port=config.port, log_level=config.log_level.lower())


def _get_streamable_http_app(mcp):
    """Return FastMCP's underlying Starlette ASGI app for the streamable-http transport.

    Tries the FastMCP 3.x ``http_app`` API first; falls back to the older
    ``streamable_http_app`` if that's what the installed version exposes.
    """
    factory = getattr(mcp, "http_app", None)
    if callable(factory):
        try:
            return factory(transport="streamable-http")
        except TypeError:
            return factory()
    factory = getattr(mcp, "streamable_http_app", None)
    if callable(factory):
        return factory()
    raise RuntimeError(
        "FastMCP instance does not expose http_app() or streamable_http_app(); "
        "cannot attach auth middleware. Check fastmcp version."
    )


if __name__ == "__main__":
    main()
