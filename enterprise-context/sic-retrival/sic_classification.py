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

# Memgraph connection configuration (localhost defaults)
MEMGRAPH_HOST = "localhost"
MEMGRAPH_PORT = 7687
MEMGRAPH_USERNAME = ""
MEMGRAPH_PASSWORD = ""
MEMGRAPH_DATABASE = "memgraph"

# Configure logging
logger = logger_init("mcp-memgraph-sic")

# Initialize FastMCP server
mcp = FastMCP("mcp-memgraph-sic")

# Initialize Memgraph client
logger.info(
    "Connecting to Memgraph db '%s' at %s:%s",
    MEMGRAPH_DATABASE,
    MEMGRAPH_HOST,
    MEMGRAPH_PORT,
)

db = Memgraph(
    host=MEMGRAPH_HOST,
    port=MEMGRAPH_PORT,
    username=MEMGRAPH_USERNAME,
    password=MEMGRAPH_PASSWORD,
    database=MEMGRAPH_DATABASE,
)

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


async def generate_clarifying_facts(
    prompt: str,
    candidates: List[Dict[str, Any]],
    ctx: Context,
) -> Dict[str, Any]:
    """
    Generate 3 clarifying facts for the user to choose from when confidence is low.

    Args:
        prompt: The user's business description
        candidates: List of candidate SIC entries with context
        ctx: FastMCP context for sampling

    Returns:
        Dictionary with facts and initial analysis
    """
    # Format candidates for the LLM
    candidates_text = _format_candidates_for_prompt(candidates)

    analysis_prompt = f"""You are an expert in SIC (Standard Industrial Classification) codes.

A user has described their business activity as follows:
"{prompt}"

Based on vector similarity search, here are the top candidate SIC classifications:
{candidates_text}

The description is ambiguous or could match multiple SIC codes. Generate exactly 3 clarifying statements that would help distinguish between the possible classifications.

Each fact should be a simple statement that the user can confirm or deny about their business.

Return your response as JSON with this structure:
{{
    "fact_1": "Your business primarily involves [specific activity A]",
    "fact_2": "Your business primarily involves [specific activity B]",
    "fact_3": "Your business primarily involves [specific activity C]",
    "reasoning": "Brief explanation of why these facts help distinguish between candidates"
}}

IMPORTANT:
- Each fact should clearly map to one of the candidate SIC codes
- Facts should be mutually exclusive where possible
- Use simple, clear language the user can easily understand
"""

    try:
        response = await ctx.sample(
            messages=analysis_prompt,
            system_prompt=(
                "You are a SIC classification expert. Generate clarifying facts "
                "to help distinguish between possible SIC codes. "
                "Return only valid JSON, no additional text or markdown."
            ),
            temperature=0.3,
            max_tokens=500,
        )

        response_text = _extract_response_text(response)
        result = json.loads(response_text)
        return result

    except json.JSONDecodeError as e:
        logger.error("Failed to parse clarifying facts response: %s", str(e))
        return {"error": "Failed to generate clarifying facts"}
    except Exception as e:
        logger.error("Clarifying facts generation failed: %s", str(e))
        return {"error": f"Failed to generate facts: {str(e)}"}


async def analyze_with_user_selection(
    prompt: str,
    candidates: List[Dict[str, Any]],
    selected_fact: str,
    ctx: Context,
) -> Dict[str, Any]:
    """
    Final analysis after user has selected a clarifying fact.

    Args:
        prompt: The user's business description
        candidates: List of candidate SIC entries with context
        selected_fact: The fact the user confirmed
        ctx: FastMCP context for sampling

    Returns:
        Dictionary with final SIC code selection
    """
    candidates_text = _format_candidates_for_prompt(candidates)

    analysis_prompt = f"""You are an expert in SIC (Standard Industrial Classification) codes.

A user has described their business activity as follows:
"{prompt}"

The user has confirmed the following fact about their business:
"{selected_fact}"

Based on vector similarity search, here are the candidate SIC classifications:
{candidates_text}

Given the user's confirmation, select the BEST matching SIC code.

Return your response as JSON with this structure:
{{
    "selected_code": "XXXX",
    "selected_name": "Name of the selected classification",
    "confidence": "high",
    "explanation": "Detailed explanation of why this code was selected based on the confirmed fact"
}}

IMPORTANT:
- The confidence should now be "high" since the user has clarified their business
- Use the most specific code possible (4-digit Industry code if available)
"""

    try:
        response = await ctx.sample(
            messages=analysis_prompt,
            system_prompt=(
                "You are a SIC classification expert. Select the best SIC code "
                "based on the user's confirmed business activity. "
                "Return only valid JSON, no additional text or markdown."
            ),
            temperature=0.1,
            max_tokens=500,
        )

        response_text = _extract_response_text(response)
        result = json.loads(response_text)
        return result

    except json.JSONDecodeError as e:
        logger.error("Failed to parse final analysis response: %s", str(e))
        return {"error": "Failed to parse final classification"}
    except Exception as e:
        logger.error("Final analysis failed: %s", str(e))
        return {"error": f"Final classification failed: {str(e)}"}


def _format_candidates_for_prompt(candidates: List[Dict[str, Any]]) -> str:
    """Format candidates list into text for LLM prompts."""
    candidates_text = ""
    for i, candidate in enumerate(candidates, 1):
        candidates_text += f"\n--- Candidate {i} ---\n"
        candidates_text += f"Industry Group Code: {candidate.get('code', 'N/A')}\n"
        candidates_text += f"Industry Group Name: {candidate.get('name', 'N/A')}\n"
        candidates_text += f"Context: {candidate.get('context_text', 'N/A')}\n"

        if "related_industries" in candidate:
            candidates_text += "Related Industries:\n"
            for industry in candidate["related_industries"][:5]:
                ind_code = industry.get("code", "")
                ind_name = industry.get("name", "")
                ind_desc = industry.get("description", "")
                candidates_text += f"  - {ind_code}: {ind_name} - {ind_desc}\n"

        if "major_group" in candidate:
            mg = candidate["major_group"]
            mg_code = mg.get("code", "")
            mg_name = mg.get("name", "")
            mg_desc = mg.get("description", "")
            candidates_text += f"Major Group: {mg_code}: {mg_name} - {mg_desc}\n"

    return candidates_text


def _extract_response_text(response) -> str:
    """Extract text from various response types."""
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

    return response_text


async def analyze_and_select_sic_code(
    prompt: str,
    candidates: List[Dict[str, Any]],
    ctx: Context,
) -> Dict[str, Any]:
    """
    Use LLM sampling to analyze candidates and select the best SIC code.
    Returns confidence as either "high" or "low".

    Args:
        prompt: The user's business description
        candidates: List of candidate SIC entries with context
        ctx: FastMCP context for sampling

    Returns:
        Dictionary with selected SIC code and explanation
    """
    candidates_text = _format_candidates_for_prompt(candidates)

    analysis_prompt = f"""You are an expert in SIC (Standard Industrial Classification) codes.

A user has described their business activity as follows:
"{prompt}"

Based on vector similarity search, here are the top candidate SIC classifications:
{candidates_text}

Your task:
1. Analyze how well each candidate matches the user's business description
2. Select the BEST matching SIC code (4-digit code from Industries if specific match, or Industry Group code if more general)
3. Determine if you are confident in this match

Return your response as JSON with this structure:
{{
    "selected_code": "XXXX",
    "selected_name": "Name of the selected classification",
    "confidence": "high" or "low",
    "explanation": "Detailed explanation of why this code was selected"
}}

CONFIDENCE RULES:
- "high": The user's description clearly and unambiguously matches one SIC code
- "low": The description is vague, ambiguous, or could match multiple SIC codes

IMPORTANT:
- Use the most specific code possible (4-digit Industry code if available)
- Be conservative - if there's any ambiguity, use "low" confidence
- Consider the full context including related industries and major groups
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
            max_tokens=500,
        )

        response_text = _extract_response_text(response)
        result = json.loads(response_text)

        # Normalize confidence to only high/low
        conf = result.get("confidence", "low").lower()
        result["confidence"] = "high" if conf == "high" else "low"

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

    If the initial classification has low confidence, the tool will ask
    clarifying questions to better understand the business activity.

    Args:
        prompt: A description of the business activity (e.g., "I work with private
                citizens in a private bank providing them commercial credits")
        ctx: FastMCP context for sampling

    Returns:
        Dictionary containing:
        - selected_code: The best matching SIC code
        - selected_name: Name of the classification
        - confidence: Confidence level (high/low)
        - explanation: Why this code was selected
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
        mg_code = None
        if candidate["major_group"]:
            mg_code = candidate["major_group"].get("code")
        logger.info(
            "  Context for %s: %d related industries, major_group=%s",
            code,
            len(candidate["related_industries"]),
            mg_code,
        )
        for ind in candidate["related_industries"]:
            logger.info(
                "    Industry: %s - %s",
                ind.get("code", ""),
                ind.get("name", ""),
            )

    # Step 4: Initial LLM analysis
    logger.info("Performing initial analysis with LLM...")
    analysis_result = await analyze_and_select_sic_code(prompt, candidates, ctx)

    # Check for errors
    if "error" in analysis_result:
        analysis_result["candidates"] = candidates
        analysis_result["prompt"] = prompt
        return analysis_result

    # Step 5: If confidence is high, return immediately
    if analysis_result.get("confidence") == "high":
        logger.info("High confidence result, returning immediately")
        analysis_result["candidates"] = candidates
        analysis_result["prompt"] = prompt
        return analysis_result

    # Step 6: Low confidence - generate clarifying facts
    logger.info("Low confidence, generating clarifying facts...")
    facts_result = await generate_clarifying_facts(prompt, candidates, ctx)

    if "error" in facts_result:
        # Fallback to initial result if fact generation fails
        logger.warning("Failed to generate facts, returning initial result")
        analysis_result["candidates"] = candidates
        analysis_result["prompt"] = prompt
        analysis_result["clarification_failed"] = True
        return analysis_result

    # Step 7: Elicit user to select a fact
    fact_1 = facts_result.get("fact_1", "Option 1")
    fact_2 = facts_result.get("fact_2", "Option 2")
    fact_3 = facts_result.get("fact_3", "Option 3")

    elicit_message = (
        "To better classify your business, please select the statement "
        "that best describes your primary activity:\n\n"
        f"1. {fact_1}\n\n"
        f"2. {fact_2}\n\n"
        f"3. {fact_3}"
    )

    logger.info("Eliciting user selection...")

    try:
        elicit_result = await ctx.elicit(
            message=elicit_message,
            response_type=["1", "2", "3"],
        )

        if elicit_result.action == "accept":
            selected_option = elicit_result.data
            logger.info("User selected option: %s", selected_option)

            # Map selection to the fact
            if selected_option == "1":
                selected_fact = fact_1
            elif selected_option == "2":
                selected_fact = fact_2
            else:
                selected_fact = fact_3

            # Step 8: Final analysis with user's selection
            logger.info("Performing final analysis with user selection...")
            final_result = await analyze_with_user_selection(
                prompt, candidates, selected_fact, ctx
            )

            final_result["candidates"] = candidates
            final_result["prompt"] = prompt
            final_result["user_clarification"] = selected_fact
            return final_result

        elif elicit_result.action == "decline":
            logger.info("User declined clarification")
            analysis_result["candidates"] = candidates
            analysis_result["prompt"] = prompt
            analysis_result["user_declined_clarification"] = True
            return analysis_result

        else:  # cancel
            logger.info("User cancelled")
            return {
                "status": "cancelled",
                "message": "Classification cancelled by user",
                "prompt": prompt,
            }

    except Exception as e:
        logger.error("Elicitation failed: %s", str(e))
        # Fallback to initial low-confidence result
        analysis_result["candidates"] = candidates
        analysis_result["prompt"] = prompt
        analysis_result["elicitation_error"] = str(e)
        return analysis_result


logger.info("🏭 SIC Classification MCP server initialized")
logger.info("Available tools: get_sic")
