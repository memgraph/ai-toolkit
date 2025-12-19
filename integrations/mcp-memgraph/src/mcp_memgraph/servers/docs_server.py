from fastmcp import FastMCP
from fastmcp.utilities.types import Image

from memgraph_toolbox.api.memgraph import Memgraph
from memgraph_toolbox.tools.cypher import CypherTool
from memgraph_toolbox.utils.logger import logger_init

from typing import Any, Dict, List, Union
import re
import httpx
from bs4 import BeautifulSoup
from urllib.parse import urljoin

from mcp_memgraph.config import get_memgraph_config, get_mcp_config

# Get configuration instances
memgraph_config = get_memgraph_config()
mcp_config = get_mcp_config()

# Configure logging
logger = logger_init("mcp-memgraph-docs")

# Initialize FastMCP server
mcp = FastMCP("mcp-memgraph-docs")

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
    for pattern in WRITE_PATTERNS:
        if re.search(pattern, query_upper):
            return True
    return False


# Initialize Memgraph client using configuration
logger.info(
    "Memgraph connection configured for db '%s' at %s with user '%s'",
    memgraph_config.database,
    memgraph_config.url,
    memgraph_config.username,
)
logger.info("Read-only mode: %s", READ_ONLY_MODE)

# Lazy initialization - connection will be established on first use
db = None


def get_db():
    """Get or create Memgraph database connection."""
    global db
    if db is None:
        try:
            db = Memgraph(**memgraph_config.get_client_config())
            logger.info("Successfully connected to Memgraph")
        except Exception as e:
            logger.error("Failed to connect to Memgraph: %s", str(e))
            raise
    return db


@mcp.tool()
def run_query(query: str) -> List[Dict[str, Any]]:
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

    try:
        result = CypherTool(db=get_db()).call({"query": query})
        return result
    except Exception as e:
        return [{"error": f"Error running query: {str(e)}"}]


@mcp.tool()
def search_relevant_documents(question: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Search for relevant document chunks based on your question.

    Use this tool when you need to find information in the documentation. It understands
    the meaning of your question and returns the most relevant text passages that can
    help answer it.

    Args:
        question: Your question or topic you want to find information about
        limit: Maximum number of document chunks to return (default: 10, recommended to use 10 or more)

    Returns:
        List of relevant document chunks with similarity scores
    """
    logger.info("Performing vector search for: %s (limit: %d)", question, limit)

    # Sanitize the question to prevent Cypher injection
    sanitized_query = question.replace("'", "\\'").replace("\\", "\\\\")

    # Construct the vector search query
    cypher_query = f"""CALL embeddings.text(["{sanitized_query}"]) yield dimension, embeddings, success
WITH embeddings[0] as embedding
CALL vector_search.search("vs_name", {limit}, embedding) YIELD distance, node, similarity
RETURN node, similarity"""

    try:
        result = CypherTool(db=get_db()).call({"query": cypher_query})

        # Remove embedding fields to save tokens
        def remove_embeddings(obj):
            if isinstance(obj, dict):
                return {
                    k: remove_embeddings(v) for k, v in obj.items() if k != "embedding"
                }
            elif isinstance(obj, list):
                return [remove_embeddings(item) for item in obj]
            else:
                return obj

        cleaned_result = remove_embeddings(result)
        return cleaned_result
    except Exception as e:
        logger.error("Error executing vector search: %s", str(e))
        return [{"error": f"Error executing vector search: {str(e)}"}]


@mcp.tool()
def search_documentation_for_keyword(
    search_term: str, limit: int = 10
) -> List[Dict[str, Any]]:
    """Search documentation by specific keywords or phrases.

    Use this tool when you want to find exact matches for specific terms, names, or phrases
    in the documentation. This is useful when you know the exact keyword you're looking for.

    Args:
        search_term: The specific keyword or phrase to find
        limit: Maximum number of results to return (default: 10, recommended to use 10 or more)

    Returns:
        List of matching document chunks with relevance scores
    """
    logger.info("Keyword search for term '%s' (limit: %d)", search_term, limit)

    # Sanitize inputs to prevent Cypher injection
    sanitized_term = search_term.replace("'", "\\'").replace("\\", "\\\\")

    # Construct the keyword search query with hardcoded "entity_id" property
    cypher_query = f"""CALL text_search.search_all("entity_id", "{sanitized_term}") YIELD node, score
RETURN node, score
ORDER BY score DESC
LIMIT {limit}"""

    try:
        result = CypherTool(db=get_db()).call({"query": cypher_query})
        return result
    except Exception as e:
        logger.error("Error executing keyword search: %s", str(e))
        return [{"error": f"Error executing keyword search: {str(e)}"}]


@mcp.tool()
def search_cypher_queries_for_keyword(
    search_term: str, limit: int = 10
) -> List[Dict[str, Any]]:
    """Search for queries related to a specific concept or topic.

    Use this tool when you want to find what questions or queries have been asked
    about a particular concept. This helps you discover different ways people ask
    about a topic or find related query patterns.

    Args:
        search_term: The concept or topic to find queries about
        limit: Maximum number of queries to return (default: 10, recommended to use 10 or more)

    Returns:
        List of related queries with their relevance scores
    """
    logger.info("Searching queries for concept '%s' (limit: %d)", search_term, limit)

    # Sanitize inputs to prevent Cypher injection
    sanitized_term = search_term.replace("'", "\\'").replace("\\", "\\\\")

    # Construct the query search
    cypher_query = f"""CALL text_search.search_all("query", "{sanitized_term}") YIELD node, score
RETURN node, score
ORDER BY score DESC
LIMIT {limit}"""

    try:
        result = CypherTool(db=get_db()).call({"query": cypher_query})
        return result
    except Exception as e:
        logger.error("Error executing query search: %s", str(e))
        return [{"error": f"Error executing query search: {str(e)}"}]


@mcp.tool()
def get_relevant_urls_for_search_term(
    search_term: str, limit: int = 10
) -> List[Dict[str, Any]]:
    """Search for relevant URLs based on a search term.

    Use this tool when you want to find URLs that are relevant to a specific topic or keyword.
    This searches through URL descriptions to find the most relevant documentation pages.

    Args:
        search_term: The keyword or phrase to search for in URL descriptions
        limit: Maximum number of URLs to return (default: 10)

    Returns:
        List of relevant URLs with their descriptions and relevance scores
    """
    logger.info("Searching URLs for term '%s' (limit: %d)", search_term, limit)

    # Sanitize inputs to prevent Cypher injection
    sanitized_term = search_term.replace("'", "\\'").replace("\\", "\\\\")

    # Construct the URL search query
    cypher_query = f"""CALL text_search.search_all("url_description", "{sanitized_term}") YIELD node, score
RETURN node.name as url, node.description as description, score
ORDER BY score DESC
LIMIT {limit}"""

    try:
        result = CypherTool(db=get_db()).call({"query": cypher_query})
        return result
    except Exception as e:
        logger.error("Error executing URL search: %s", str(e))
        return [{"error": f"Error executing URL search: {str(e)}"}]


@mcp.tool()
def read_url(url: str) -> List[Dict[str, Any]]:
    """Read and retrieve content from a URL on the internet.

    Use this tool when you find a URL and want to read its actual content from the web.
    This fetches the webpage, extracts the text content, and returns it along with all
    URLs and image URLs found on the page for navigation. Use the get_image tool to
    retrieve actual image data in base64 format. Useful for accessing external
    documentation, references, or resources.

    Args:
        url: The URL to read content from

    Returns:
        The text content extracted from the URL, a list of URLs found on the page,
        and a list of image URLs with alt text
    """
    logger.info("Fetching content from URL: %s", url)

    try:
        # Fetch the URL content
        with httpx.Client(timeout=10.0) as client:
            response = client.get(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; MCP-Memgraph-Docs/1.0)"
                },
            )
            response.raise_for_status()

            # Parse HTML content
            soup = BeautifulSoup(response.content, "html.parser")

            # Special handling for memgraph.com/docs - find main inside main
            if "memgraph.com/docs" in url:
                # Find outer <main> element
                outer_main = soup.find("main")
                if not outer_main:
                    raise ValueError(
                        "Could not find outer <main> element on memgraph.com/docs page"
                    )

                # Find inner <main> element within the outer one
                inner_main = outer_main.find("main")
                if not inner_main:
                    raise ValueError(
                        "Could not find inner <main> element within outer <main> on memgraph.com/docs page"
                    )

                content_element = inner_main
            else:
                content_element = soup

            # Extract all URLs from the content element
            page_urls = []
            for link in content_element.find_all("a", href=True):
                href = link["href"]
                # Convert relative URLs to absolute
                if href.startswith("/"):
                    href = urljoin(url, href)
                if href.startswith("http"):
                    page_urls.append(
                        {"url": href, "text": link.get_text(strip=True) or "No text"}
                    )

            # Extract image URLs from the content element only
            images = []
            for img in content_element.find_all("img", src=True):
                img_url = img["src"]
                # Convert relative URLs to absolute
                if not img_url.startswith("http"):
                    img_url = urljoin(url, img_url)

                images.append({"url": img_url, "alt": img.get("alt", "")})

            # Remove script and style elements
            for script in content_element(
                ["script", "style", "nav", "footer", "header"]
            ):
                script.decompose()

            # Get text content
            text = content_element.get_text(separator="\n", strip=True)

            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            text = "\n".join(line for line in lines if line)

            return [
                {
                    "url": url,
                    "content": text[
                        :50000
                    ],  # Limit to 50k characters to avoid token overflow
                    "page_urls": page_urls[
                        :100
                    ],  # Limit to 100 URLs to avoid token overflow
                    "images": images[:50],  # Limit to 50 images to avoid excessive data
                    "status": "success",
                }
            ]

    except httpx.TimeoutException:
        logger.error("Timeout while fetching URL: %s", url)
        return [{"error": f"Timeout while fetching URL: {url}"}]
    except httpx.HTTPStatusError as e:
        logger.error("HTTP error fetching URL %s: %s", url, str(e))
        return [{"error": f"HTTP error fetching URL: {str(e)}"}]
    except httpx.RequestError as e:
        logger.error("Error fetching URL %s: %s", url, str(e))
        return [{"error": f"Error fetching URL: {str(e)}"}]
    except Exception as e:
        logger.error("Error processing URL %s: %s", url, str(e))
        return [{"error": f"Error processing URL: {str(e)}"}]


@mcp.tool()
def get_image(image_url: str) ->  Union[Image, List[Dict[str, Any]]]:
    """Retrieve an image from a URL and return a renderable format of it.

    Use this tool when you have an image URL (e.g., from read_url results) and want
    to retrieve the actual image data.
    When you call the `read_url` tool, it will also display the urls of images on the page
    you can use in order to augment the response with. It's always good to get the images in order
    to have a more interactive response with the user.

    Args:
        image_url: The URL of the image to retrieve

    Returns:
        Image format that can be displayed to the user.
    """
    logger.info("Fetching image from URL: %s", image_url)
    return f"![]({image_url})"

    # try:
    #     with httpx.Client(timeout=10.0) as client:
    #         response = client.get(
    #             image_url,
    #             headers={
    #                 "User-Agent": "Mozilla/5.0 (compatible; MCP-Memgraph-Docs/1.0)"
    #             },
    #         )
    #         response.raise_for_status()

    #         # Get content type from response headers
    #         content_type = response.headers.get("content-type", "png").replace("image/", "")

    #         return Image(data=response.content, format=content_type)

    # except httpx.TimeoutException:
    #     logger.error("Timeout while fetching image: %s", image_url)
    #     return [{"error": f"Timeout while fetching image: {image_url}"}]
    # except httpx.HTTPStatusError as e:
    #     logger.error("HTTP error fetching image %s: %s", image_url, str(e))
    #     return [{"error": f"HTTP error fetching image: {str(e)}"}]
    # except httpx.RequestError as e:
    #     logger.error("Error fetching image %s: %s", image_url, str(e))
    #     return [{"error": f"Error fetching image: {str(e)}"}]
    # except Exception as e:
    #     logger.error("Error processing image %s: %s", image_url, str(e))
    #     return [{"error": f"Error processing image: {str(e)}"}]

@mcp.tool()
def read_connected_information(node_id: int) -> List[Dict[str, Any]]:
    """Read all information connected to a node you already found.

    Use this tool when you have found something relevant (a document chunk, entity,
    or any other node) and want to explore what's connected to it. This reveals
    related information, surrounding context, and connected entities that can help
    you build a more complete understanding.

    Args:
        node_id: The ID of the node you want to explore connections from

    Returns:
        The center node and all its connected neighbors with their relationships
    """
    logger.info("Reading connected information for node ID: %d", node_id)

    # Construct the relevance expansion query
    cypher_query = f"""MATCH (n)-[r]-(m)
WHERE n.node_id = {node_id}
RETURN n as center_node,
       collect(DISTINCT {{relationship: r, neighbor: m}}) as connections"""

    try:
        result = CypherTool(db=get_db()).call({"query": cypher_query})
        return result
    except Exception as e:
        logger.error("Error executing relevance expansion: %s", str(e))
        return [{"error": f"Error executing relevance expansion: {str(e)}"}]
