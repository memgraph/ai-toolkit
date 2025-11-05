from mcp_memgraph.server import mcp, logger
from mcp_memgraph.config import get_mcp_config
from typing import Literal, cast


def main():
    # Get MCP server configuration
    config = get_mcp_config()

    logger.info(
        f"Starting MCP server on {config.host} with transport: {config.transport}"
    )

    # Run server with configuration
    # Note: host parameter is only used for HTTP/SSE transports, not stdio
    transport = cast(Literal["stdio", "streamable-http"], config.transport)

    if config.transport == "stdio":
        mcp.run(transport=transport)
    else:
        mcp.run(host=config.host, transport=transport)


if __name__ == "__main__":
    main()
