from mcp.server.fastmcp import FastMCP
from neo4j import GraphDatabase
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp-memgraph")

# Initialize FastMCP server
mcp = FastMCP("mcp-memgraph")

MEMGRAPH_URI = "bolt://localhost:7687"
MEMGRAPH_USER = ""
MEMGRAPH_PASSWORD = ""

driver = GraphDatabase.driver(MEMGRAPH_URI, auth=(MEMGRAPH_USER, MEMGRAPH_PASSWORD))


def execute_query(query):
    """Helper function to execute a query on Memgraph"""
    with driver.session() as session:
        return session.run(query).data()


@mcp.tool()
def run_query(query: str) -> list:
    """Run a query against Memgraph"""
    logger.info(f"Running query: {query}")
    try:
        result = execute_query(query)
        return result
    except Exception as e:
        return [f"Error: {str(e)}"]


# Can be used as a tool as well
# @mcp.tool()
# def get_schema() -> str:
#     """Get Memgraph schema information"""
#     logger.info("Fetching Memgraph schema...")
#     try:
#         result = execute_query("SHOW SCHEMA INFO;")
#         return result
#     except Exception as e:
#         return f"Error fetching schema: {str(e)}"


# If used as a resource, it needs to be fetched before (not with helper function)
@mcp.resource("schema://main")
def get_schema() -> str:
    """Get Memgraph schema information as a resouce"""
    logger.info("Fetching Memgraph schema...")
    try:
        driver = GraphDatabase.driver(
            MEMGRAPH_URI, auth=(MEMGRAPH_USER, MEMGRAPH_PASSWORD)
        )
        with driver.session() as session:
            return session.run("SHOW SCHEMA INFO").data()
    except Exception as e:
        return f"Error fetching schema: {str(e)}"


if __name__ == "__main__":
    # Initialize and run the server
    logger.info("Starting FastMCP server...")
    mcp.run(transport="stdio")
