import os
from mcp_memgraph.config import get_mcp_config
from mcp_memgraph.servers import get_server, get_server_info
from typing import Literal, cast


def main():
    # Get MCP server configuration
    config = get_mcp_config()

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
    transport = cast(Literal["stdio", "streamable-http"], config.transport)

    if config.transport == "stdio":
        mcp.run(transport=transport)
    else:
        mcp.run(host=config.host, transport=transport)


if __name__ == "__main__":
    main()
