"""SIC Classification MCP Server.

This server provides SIC (Standard Industrial Classification) code lookup
capabilities using vector search in Memgraph.

The server uses:
- Vector search to find relevant IndustryGroup nodes based on user description
- Node neighborhood exploration to gather context (Industries, MajorGroups)
- LLM sampling to determine the best matching SIC code
"""

from fastmcp import FastMCP, Context
from memgraph_toolbox.api.memgraph import Memgraph
from memgraph_toolbox.tools.node_vector_search import NodeVectorSearchTool
from memgraph_toolbox.tools.node_neighborhood import NodeNeighborhoodTool
from memgraph_toolbox.utils.logger import logger_init

from typing import Any, Dict, List
import json

from mcp_memgraph.config import get_memgraph_config, get_mcp_config

# Get configuration instances
memgraph_config = get_memgraph_config()
mcp_config = get_mcp_config()

# Configure logging
logger = logger_init("mcp-memgraph-sic")

# Initialize FastMCP server
mcp = FastMCP("mcp-memgraph-sic")

# Initialize Memgraph client
logger.info(
    "Connecting to Memgraph db '%s' at %s with user '%s'",
    memgraph_config.database,
    memgraph_config.url,
    memgraph_config.username,
)

db = Memgraph(**memgraph_config.get_client_config())

# Vector index configuration
SIC_VECTOR_INDEX = "sic_industry_group_embedding"
EMBEDDING_DIMENSION = 384
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Lazy-loaded embedding model
_embedding_model = None


def get_embedding_model():
    """Get or initialize the sentence transformer model (lazy loading)."""
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
            _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            logger.info("Embedding model loaded successfully")
        except ImportError:
            logger.error(
                "sentence_transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            raise
    return _embedding_model


def get_embedding_from_text(text: str) -> List[float]:
    """
    Generate an embedding vector for the given text using sentence transformers.

    Args:
        text: The text to embed

    Returns:
        List of floats representing the embedding vector
    """
    model = get_embedding_model()
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()


def get_node_context(node_id: int, max_distance: int = 1) -> List[Dict[str, Any]]:
    """
    Get the neighborhood context around a node.

    Args:
        node_id: The ID of the node
        max_distance: Maximum hops to traverse

    Returns:
        List of neighboring node properties
    """
    try:
        results = NodeNeighborhoodTool(db=db).call(
            {"node_id": str(node_id), "max_distance": max_distance, "limit": 50}
        )
        return results
    except Exception as e:
        logger.error("Failed to get node neighborhood: %s", str(e))
        return []


def perform_vector_search(
    query_vector: List[float], limit: int = 3
) -> List[Dict[str, Any]]:
    """
    Perform vector search on the SIC index.

    Args:
        query_vector: The embedding vector to search with
        limit: Number of results to return

    Returns:
        List of matching nodes with their properties and distances
    """
    try:
        results = NodeVectorSearchTool(db=db).call(
            {
                "index_name": SIC_VECTOR_INDEX,
                "query_vector": query_vector,
                "limit": limit,
            }
        )
        return results
    except Exception as e:
        logger.error("Vector search failed: %s", str(e))
        return []


def get_node_id_by_code(code: str, label: str = "IndustryGroup") -> int | None:
    """
    Get the internal node ID by the SIC code.

    Args:
        code: The SIC code
        label: The node label

    Returns:
        The node ID or None if not found
    """
    try:
        query = f"MATCH (n:{label} {{code: $code}}) RETURN id(n) as node_id"
        results = db.query(query, {"code": code})
        if results:
            return results[0]["node_id"]
        return None
    except Exception as e:
        logger.error("Failed to get node ID: %s", str(e))
        return None


async def analyze_and_select_sic_code(
    prompt: str,
    candidates: List[Dict[str, Any]],
    ctx: Context,
) -> Dict[str, Any]:
    """
    Use LLM sampling to analyze candidates and select the best SIC code.

    Args:
        prompt: The user's business description
        candidates: List of candidate SIC entries with context
        ctx: FastMCP context for sampling

    Returns:
        Dictionary with selected SIC code and explanation
    """
    # Format candidates for the LLM
    candidates_text = ""
    for i, candidate in enumerate(candidates, 1):
        candidates_text += f"\n--- Candidate {i} ---\n"
        candidates_text += f"Industry Group Code: {candidate.get('code', 'N/A')}\n"
        candidates_text += f"Industry Group Name: {candidate.get('name', 'N/A')}\n"
        candidates_text += f"Context: {candidate.get('context_text', 'N/A')}\n"

        # Add related industries if available
        if "related_industries" in candidate:
            candidates_text += "Related Industries:\n"
            for industry in candidate["related_industries"][:5]:  # Limit to 5
                candidates_text += f"  - {industry.get('code', '')}: {industry.get('name', '')} - {industry.get('description', '')}\n"

        # Add parent major group if available
        if "major_group" in candidate:
            mg = candidate["major_group"]
            candidates_text += f"Major Group: {mg.get('code', '')}: {mg.get('name', '')} - {mg.get('description', '')}\n"

    analysis_prompt = f"""You are an expert in SIC (Standard Industrial Classification) codes.

A user has described their business activity as follows:
"{prompt}"

Based on vector similarity search, here are the top candidate SIC classifications:
{candidates_text}

Your task:
1. Analyze how well each candidate matches the user's business description
2. Select the BEST matching SIC code (4-digit code from Industries if specific match, or Industry Group code if more general)
3. Explain why this is the best match

Return your response as JSON with this structure:
{{
    "selected_code": "XXXX",
    "selected_name": "Name of the selected classification",
    "confidence": "high|medium|low",
    "explanation": "Detailed explanation of why this code was selected",
    "alternative_codes": ["YYYY", "ZZZZ"],
    "alternative_reasons": "Brief explanation of alternatives if confidence is not high"
}}

IMPORTANT: 
- Use the most specific code possible (4-digit Industry code if available and appropriate)
- Consider the full context including related industries and major groups
- Be precise - match the actual business activity, not just keywords
"""

    try:
        response = await ctx.sample(
            messages=analysis_prompt,
            system_prompt=(
                "You are a SIC classification expert. Analyze business descriptions "
                "and match them to the most appropriate SIC code. "
                "Return only valid JSON, no additional text or markdown."
            ),
            temperature=0.2,
            max_tokens=1000,
        )

        # Parse the response
        if isinstance(response, str):
            response_text = response.strip()
        elif hasattr(response, "text"):
            response_text = response.text.strip()
        else:
            response_text = str(response).strip()

        # Remove markdown code blocks if present
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            response_text = "\n".join(
                line
                for line in lines
                if not line.strip().startswith("```") and line.strip().lower() != "json"
            ).strip()

        result = json.loads(response_text)
        return result

    except json.JSONDecodeError as e:
        logger.error("Failed to parse LLM response: %s", str(e))
        return {
            "error": "Failed to parse classification response",
            "raw_response": response_text if "response_text" in locals() else None,
        }
    except Exception as e:
        logger.error("Classification analysis failed: %s", str(e))
        return {"error": f"Classification failed: {str(e)}"}


@mcp.tool()
async def get_sic(prompt: str, ctx: Context) -> Dict[str, Any]:
    """
    Get the SIC (Standard Industrial Classification) code based on a business description.

    This tool performs a vector search on the SIC taxonomy stored in Memgraph,
    retrieves context about matching industry groups, and uses AI to determine
    the most appropriate SIC code for the described business activity.

    Args:
        prompt: A description of the business activity (e.g., "I work with private
                citizens in a private bank providing them commercial credits")
        ctx: FastMCP context for sampling

    Returns:
        Dictionary containing:
        - selected_code: The best matching SIC code
        - selected_name: Name of the classification
        - confidence: Confidence level (high/medium/low)
        - explanation: Why this code was selected
        - alternative_codes: Other potential matches
        - candidates: The raw candidate data used for selection
    """
    logger.info("get_sic called with prompt: %s", prompt)

    # Step 1: Generate embedding for the user's prompt
    logger.info("Generating embedding for prompt...")
    query_embedding = get_embedding_from_text(prompt)

    # Step 2: Perform vector search to find top 3 matching IndustryGroups
    logger.info("Performing vector search...")
    search_results = perform_vector_search(query_embedding, limit=3)

    # Log vector search results
    logger.info("Vector search returned %d results", len(search_results))
    for i, result in enumerate(search_results):
        props = result.get("properties", {})
        logger.info(
            "  Candidate %d: code=%s, name=%s, distance=%s",
            i + 1,
            props.get("code", "N/A"),
            props.get("name", "N/A"),
            result.get("distance", "N/A"),
        )

    if not search_results:
        return {
            "error": "No matching SIC classifications found",
            "prompt": prompt,
        }

    if "error" in search_results[0]:
        return {
            "error": "Vector search failed",
            "details": search_results[0].get("error"),
            "prompt": prompt,
        }

    # Step 3: For each result, get the 1-hop neighborhood context
    logger.info("Gathering context for %d candidates...", len(search_results))
    candidates = []

    for result in search_results:
        properties = result.get("properties", {})
        code = properties.get("code", "")

        candidate = {
            "code": code,
            "name": properties.get("name", ""),
            "context_text": properties.get("context_text", ""),
            "distance": result.get("distance", 0),
            "related_industries": [],
            "major_group": None,
        }

        # Get node ID and fetch neighborhood
        node_id = get_node_id_by_code(code)
        if node_id is not None:
            neighborhood = get_node_context(node_id, max_distance=1)

            for neighbor in neighborhood:
                # Classify neighbor by its properties
                if "examples" in neighbor:
                    # This is likely an Industry node
                    candidate["related_industries"].append(
                        {
                            "code": neighbor.get("code", ""),
                            "name": neighbor.get("name", ""),
                            "description": neighbor.get("description", ""),
                            "examples": neighbor.get("examples", []),
                        }
                    )
                elif "description" in neighbor and "embedding" not in neighbor:
                    # This is likely a MajorGroup node
                    candidate["major_group"] = {
                        "code": neighbor.get("code", ""),
                        "name": neighbor.get("name", ""),
                        "description": neighbor.get("description", ""),
                    }

        candidates.append(candidate)

        # Log context gathered for this candidate
        logger.info(
            "  Context for %s: %d related industries, major_group=%s",
            code,
            len(candidate["related_industries"]),
            candidate["major_group"].get("code") if candidate["major_group"] else None,
        )
        for ind in candidate["related_industries"]:
            logger.info(
                "    Industry: %s - %s",
                ind.get("code", ""),
                ind.get("name", ""),
            )

    # Step 4: Use LLM sampling to analyze and select the best SIC code
    logger.info("Analyzing candidates with LLM...")
    analysis_result = await analyze_and_select_sic_code(prompt, candidates, ctx)

    # Add the raw candidates to the result for transparency
    analysis_result["candidates"] = candidates
    analysis_result["prompt"] = prompt

    return analysis_result


logger.info("🏭 SIC Classification MCP server initialized")
logger.info("Available tools: get_sic")
