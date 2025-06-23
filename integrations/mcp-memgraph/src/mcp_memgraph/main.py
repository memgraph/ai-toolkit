import os
from mcp_memgraph.server import mcp, logger


def main():
    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    logger.info(f"Starting MCP server with transport: {transport}")
    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
