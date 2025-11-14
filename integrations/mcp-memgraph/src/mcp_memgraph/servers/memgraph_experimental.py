"""Memgraph Experimental.

This server provides autonomous adapting GraphRAG capabilities that:
- Uses sampling to validate queries and check for required indexes
- Uses elicitation to prompt users about creating missing indexes
- Automatically detects vector, text, and label indexes in Memgraph
- Adapts query execution based on available indexes
"""

from fastmcp import FastMCP, Context
from memgraph_toolbox.api.memgraph import Memgraph
from memgraph_toolbox.tools.cypher import CypherTool
from memgraph_toolbox.tools.index import ShowIndexInfoTool
from memgraph_toolbox.tools.schema import ShowSchemaInfoTool
from memgraph_toolbox.utils.logger import logger_init

from typing import Any, Dict, List
from dataclasses import dataclass
import json

from mcp_memgraph.config import get_memgraph_config, get_mcp_config

# Get configuration instances
memgraph_config = get_memgraph_config()
mcp_config = get_mcp_config()

# Configure logging
logger = logger_init("mcp-memgraph-experimental")

# Initialize FastMCP server
mcp = FastMCP("mcp-memgraph-experimental")

# Initialize Memgraph client
logger.info(
    "Connecting to Memgraph db '%s' at %s with user '%s'",
    memgraph_config.database,
    memgraph_config.url,
    memgraph_config.username,
)
logger.info("Memgraph Experimental server initialized")
logger.warning(
    "Note: Read-only mode is not supported on this server. "
    "This server requires write access to create indexes."
)

# TODO (@antejavor): Implement some tests for this
db = Memgraph(**memgraph_config.get_client_config())


# ============================================================================
# Structured Output for Sampling
# ============================================================================


@dataclass
class QueryAnalysis:
    """Structured output for query analysis via sampling."""

    labels: List[str]
    properties: List[str]
    uses_vector_search: bool
    uses_text_search: bool


# ============================================================================
# Index Detection Utilities
# ============================================================================


class IndexInfo:
    """Information about indexes in Memgraph."""

    def __init__(self):
        self.label_indexes: List[Dict[str, Any]] = []
        self.label_property_indexes: List[Dict[str, Any]] = []
        self.vector_indexes: List[Dict[str, Any]] = []
        self.text_indexes: List[Dict[str, Any]] = []

    def has_label_index(self, label: str) -> bool:
        """Check if a label-only index exists."""
        return any(idx.get("label") == label for idx in self.label_indexes)

    def has_label_property_index(self, label: str, property: str) -> bool:
        """Check if a label+property index exists."""
        return any(
            idx.get("label") == label and property in idx.get("properties", [])
            for idx in self.label_property_indexes
        )

    def has_vector_index(self, label: str, property: str) -> bool:
        """Check if a vector index exists for a label and property."""
        return any(
            idx.get("label") == label and idx.get("property") == property
            for idx in self.vector_indexes
        )

    def has_text_index(self, label: str, property: str) -> bool:
        """Check if a text index exists for a label and property."""
        return any(
            idx.get("label") == label and idx.get("property") == property
            for idx in self.text_indexes
        )


def get_current_indexes() -> IndexInfo:
    """Retrieve all current indexes from Memgraph and categorize them."""
    logger.info("Fetching current indexes from Memgraph")
    index_info = IndexInfo()

    try:
        indexes_raw = ShowIndexInfoTool(db=db).call({})

        for idx in indexes_raw:
            # Parse index information based on Memgraph's
            # SHOW INDEX INFO format
            # Format: {"index type": "...", "label": "...", "property": [...]}
            if isinstance(idx, dict):
                # Note: key is "index type" not "type"
                index_type = idx.get("index type", "").lower()
                label = idx.get("label", "")
                # Note: key is "property" not "properties"
                properties = idx.get("property", [])

                # Categorize by type
                if "vector" in index_type:
                    for prop in properties:
                        index_info.vector_indexes.append(
                            {
                                "label": label,
                                "property": prop,
                                "type": index_type,
                            }
                        )
                elif "text" in index_type:
                    for prop in properties:
                        index_info.text_indexes.append(
                            {
                                "label": label,
                                "property": prop,
                                "type": index_type,
                            }
                        )
                elif "label" in index_type and not properties:
                    # Label-only index (no properties)
                    index_info.label_indexes.append(
                        {"label": label, "type": index_type}
                    )
                elif "label+property" in index_type or "label-property" in index_type:
                    for prop in properties:
                        index_info.label_property_indexes.append(
                            {
                                "label": label,
                                "properties": [prop],
                                "type": index_type,
                            }
                        )

        logger.info(
            "Found %d label, %d label+property, %d vector, %d text indexes",
            len(index_info.label_indexes),
            len(index_info.label_property_indexes),
            len(index_info.vector_indexes),
            len(index_info.text_indexes),
        )
        logger.debug("Label indexes: %s", index_info.label_indexes)
        logger.debug("Label+property indexes: %s", index_info.label_property_indexes)
        logger.debug("Vector indexes: %s", index_info.vector_indexes)
        logger.debug("Text indexes: %s", index_info.text_indexes)

    except Exception as e:
        logger.error("Error fetching indexes: %s", str(e))

    return index_info


# ============================================================================
# Query Analysis via Sampling
# ============================================================================


async def analyze_query_with_sampling(query: str, ctx: Context) -> Dict[str, Any]:
    """
    Analyze a Cypher query using LLM sampling.

    Args:
        query: The Cypher query to analyze
        ctx: FastMCP context for sampling

    Returns:
        Dictionary with analysis results
    """
    analysis_prompt = f"""Analyze Cypher query and identify ALL node labels used, even without property filters.

Query: {query}

CRITICAL: ALWAYS include an entry for EVERY node label in the MATCH clause.
- If a label has property filters (WHERE or inline {{}}), use that property
- If a label has NO property filter, use empty string "" for property (label-only index)

For each label found:
- label: Node label (REQUIRED for every label in query)
- property: Property name if filtered, or "" for label-only
- uses_vector: true only for vector_search/embedding operations
- uses_text: true only for CONTAINS/STARTS WITH/ENDS WITH text search

Return JSON: {{"indexed_filters": [...]}}

EXAMPLES:

1. Label-only (MATCH label without property filter):
Query: MATCH (p:Person) RETURN p.name, p.age
Result: {{"indexed_filters": [{{"label": "Person", "property": "", "uses_vector": false, "uses_text": false}}]}}
Reason: Person label is used, no property filter, so property=""

2. Label-only with count:
Query: MATCH (c:City) RETURN count(c)
Result: {{"indexed_filters": [{{"label": "City", "property": "", "uses_vector": false, "uses_text": false}}]}}
Reason: City label is used, no property filter, so property=""

3. Label+property (inline filter):
Query: MATCH (c:City {{name: 'Ljubljana'}}) RETURN c.rank
Result: {{"indexed_filters": [{{"label": "City", "property": "name", "uses_vector": false, "uses_text": false}}]}}
Reason: City.name is filtered inline

4. Label+property (WHERE filter):
Query: MATCH (d:Doc) WHERE d.text CONTAINS 'word' RETURN d.title
Result: {{"indexed_filters": [{{"label": "Doc", "property": "text", "uses_vector": false, "uses_text": true}}]}}
Reason: Doc.text is filtered with CONTAINS (text search)

5. Vector index:
Query: CALL vector_search.search('idx', 10, $emb) YIELD node
Result: {{"indexed_filters": [{{"label": "Document", "property": "embedding", "uses_vector": true, "uses_text": false}}]}}
Reason: Vector search operation
"""

    try:
        # Use sampling to analyze the query
        response = await ctx.sample(
            messages=analysis_prompt,
            system_prompt=(
                "You are a Cypher index optimizer. "
                "Extract ONLY properties used in filters, not RETURN. "
                "Respond only with valid JSON, no additional text."
            ),
            temperature=0.1,
            max_tokens=500,
        )

        # Parse the JSON response - handle both string and object responses
        if isinstance(response, str):
            response_text = response.strip()
        elif hasattr(response, "text"):
            response_text = response.text.strip()
        else:
            # Try to get content from response object
            response_text = str(response).strip()

        # Remove markdown code blocks if present
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            response_text = "\n".join(
                line for line in lines if not line.strip().startswith("```")
            )

        analysis_data = json.loads(response_text)

        # Process indexed_filters from the new format
        recommended_indexes = []

        for filter_info in analysis_data.get("indexed_filters", []):
            label = filter_info.get("label", "")
            prop = filter_info.get("property", "")
            uses_vector = filter_info.get("uses_vector", False)
            uses_text = filter_info.get("uses_text", False)

            if not label:
                continue

            # Label-only index (no property specified)
            if not prop or prop == "":
                recommended_indexes.append(
                    {"label": label, "property": "", "type": "label"}
                )
            elif uses_vector:
                recommended_indexes.append(
                    {"label": label, "property": prop, "type": "vector"}
                )
            elif uses_text:
                recommended_indexes.append(
                    {"label": label, "property": prop, "type": "text"}
                )
            else:
                recommended_indexes.append(
                    {"label": label, "property": prop, "type": "label+property"}
                )

        # Extract unique labels and properties for compatibility
        labels = list(
            set(f.get("label", "") for f in analysis_data.get("indexed_filters", []))
        )
        properties = list(
            set(f.get("property", "") for f in analysis_data.get("indexed_filters", []))
        )
        uses_vector_search = any(
            f.get("uses_vector", False)
            for f in analysis_data.get("indexed_filters", [])
        )
        uses_text_search = any(
            f.get("uses_text", False) for f in analysis_data.get("indexed_filters", [])
        )

        return {
            "labels": labels,
            "properties": properties,
            "uses_vector_search": uses_vector_search,
            "uses_text_search": uses_text_search,
            "recommended_indexes": recommended_indexes,
        }

    except json.JSONDecodeError as e:
        logger.error("Failed to parse JSON from sampling response: %s", e)
        logger.error("Response was: %s", response_text)
        # Return empty analysis on parse failure
        return {
            "labels": [],
            "properties": [],
            "uses_vector_search": False,
            "uses_text_search": False,
            "recommended_indexes": [],
            "error": f"Failed to parse LLM response: {str(e)}",
        }
    except Exception as e:
        logger.error("Sampling failed: %s", str(e))
        return {
            "labels": [],
            "properties": [],
            "uses_vector_search": False,
            "uses_text_search": False,
            "recommended_indexes": [],
            "error": f"Sampling failed: {str(e)}",
        }


async def generate_index_creation_query(
    label: str, property: str, index_type: str, ctx: Context
) -> str:
    """
    Use sampling to generate the Cypher query for creating an index.

    Args:
        label: Node label
        property: Property name
        index_type: Type of index (vector, text, label+property)
        ctx: FastMCP context for sampling

    Returns:
        Cypher query string for creating the index
    """
    prompt = f"""Generate a Cypher query to create a {index_type} index.

Index details:
- Node label: {label}
- Property: {property}
- Index type: {index_type}

EXAMPLES:

1. Label+property index:
Input: label=City, property=name, type=label+property
Output: CREATE INDEX ON :City(name)

2. Text index:
Input: label=Document, property=content, type=text
Output: CREATE TEXT INDEX ON :Document(content)

3. Vector index:
Input: label=Article, property=embedding, type=vector
Output: CREATE VECTOR INDEX ON :Article(embedding) WITH CONFIG {{'dimension': 384, 'capacity': 10000}}

4. Label index (without property):
Input: label=Person, property=, type=label
Output: CREATE INDEX ON :Person

Now generate the query for the given inputs.
Respond with ONLY the Cypher query, no explanation or markdown.
"""

    try:
        response = await ctx.sample(
            messages=prompt,
            system_prompt=(
                "You are a Memgraph Cypher query expert. "
                "Generate valid index creation queries. "
                "Respond only with the query, no additional text."
            ),
            temperature=0.1,
            max_tokens=200,
        )

        # Extract query from response
        if isinstance(response, str):
            query_text = response.strip()
        elif hasattr(response, "text"):
            query_text = response.text.strip()
        else:
            query_text = str(response).strip()

        # Clean up markdown code blocks if present
        if query_text.startswith("```"):
            lines = query_text.split("\n")
            query_text = "\n".join(
                line
                for line in lines
                if not line.strip().startswith("```")
                and line.strip().lower() != "cypher"
            ).strip()

        return query_text

    except Exception as e:
        logger.error("Failed to generate index query via sampling: %s", e)
        # Fallback to manual generation
        if index_type == "label":
            return f"CREATE INDEX ON :{label}"
        elif index_type == "vector":
            return (
                f"CREATE VECTOR INDEX ON :{label}({property}) "
                "WITH CONFIG {'dimension': 384, 'capacity': 10000}"
            )
        elif index_type == "text":
            return f"CREATE TEXT INDEX ON :{label}({property})"
        else:
            return f"CREATE INDEX ON :{label}({property})"


def execute_index_creation(
    label: str, property: str, index_type: str, query: str
) -> Dict[str, Any]:
    """
    Execute the index creation query.

    Args:
        label: Node label
        property: Property name
        index_type: Type of index
        query: Cypher query to execute

    Returns:
        Dictionary with creation status
    """
    try:
        logger.info("Creating index with query: %s", query)
        db.query(query)

        return {
            "label": label,
            "property": property,
            "type": index_type,
            "status": "created",
            "query": query,
        }

    except Exception as e:
        logger.error("Error creating index: %s", str(e))
        return {
            "label": label,
            "property": property,
            "type": index_type,
            "status": "error",
            "error": str(e),
            "query": query,
        }


# ============================================================================
# MCP Tools
# ============================================================================


@mcp.tool()
async def query_tool(query: str, ctx: Context) -> List[Dict[str, Any]]:
    """
    Execute a Cypher query with intelligent index checking and
    recommendations.

    This tool uses sampling to check if appropriate indexes exist in
    Memgraph for the query being executed. If beneficial indexes are
    missing, it will use elicitation to ask the user if they want to
    create them.

    Args:
        query: The Cypher query to execute
        ctx: FastMCP context for sampling and elicitation

    Returns:
        List of query results or recommendations
    """
    logger.info("Query tool called with query: %s", query)

    # Step 1: Analyze the query using sampling
    # Use the LLM to help understand the query better
    # Use sampling to analyze the query
    analysis = await analyze_query_with_sampling(query, ctx)
    logger.info("Query analysis: %s", analysis)

    # Step 2: Get current indexes
    current_indexes = get_current_indexes()

    # Step 3: Check if recommended indexes exist
    missing_indexes = []

    for rec_idx in analysis["recommended_indexes"]:
        label = rec_idx["label"]
        prop = rec_idx["property"]
        idx_type = rec_idx["type"]

        if idx_type == "label":
            if not current_indexes.has_label_index(label):
                missing_indexes.append(rec_idx)
        elif idx_type == "vector":
            if not current_indexes.has_vector_index(label, prop):
                missing_indexes.append(rec_idx)
        elif idx_type == "text":
            if not current_indexes.has_text_index(label, prop):
                missing_indexes.append(rec_idx)
        elif idx_type == "label+property":
            if not current_indexes.has_label_property_index(label, prop):
                missing_indexes.append(rec_idx)

    # Step 4: If indexes are missing, use ELICITATION
    if missing_indexes:
        logger.info("Missing recommended indexes: %s", missing_indexes)

        # Use sampling to generate Cypher queries for creating indexes
        index_queries = []
        for idx in missing_indexes:
            cypher_query = await generate_index_creation_query(
                idx["label"], idx["property"], idx["type"], ctx
            )
            index_queries.append(
                {
                    "label": idx["label"],
                    "property": idx["property"],
                    "type": idx["type"],
                    "query": cypher_query,
                }
            )

        # Format index information for user with actual queries
        index_descriptions = []
        for idx_query in index_queries:
            if idx_query["type"] == "label":
                desc = f"Label index on {idx_query['label']}"
            elif idx_query["type"] == "vector":
                desc = (
                    f"Vector index on {idx_query['label']}." f"{idx_query['property']}"
                )
            elif idx_query["type"] == "text":
                desc = f"Text index on {idx_query['label']}." f"{idx_query['property']}"
            else:
                desc = (
                    f"Label+property index on {idx_query['label']}."
                    f"{idx_query['property']}"
                )
            desc += f"\n  Query: {idx_query['query']}"
            index_descriptions.append(desc)

        indexes_list = "\n\n".join(f"- {desc}" for desc in index_descriptions)

        # Use FastMCP's elicitation to ask user
        elicit_message = (
            "The query would benefit from these indexes for better "
            f"performance:\n\n{indexes_list}\n\n"
            "Would you like to create these indexes?"
        )

        try:
            # Request elicitation from user with Yes/No buttons
            elicit_result = await ctx.elicit(
                message=elicit_message,
                response_type=["Yes", "No"],  # Provide Yes/No button options
            )

            # Check user's response based on action
            if elicit_result.action == "accept":
                # When accepted, data contains the selected button value
                if elicit_result.data == "Yes":
                    logger.info("User approved index creation via elicitation")
                    created_indexes = []

                    for idx_query in index_queries:
                        result = execute_index_creation(
                            idx_query["label"],
                            idx_query["property"],
                            idx_query["type"],
                            idx_query["query"],
                        )
                        created_indexes.append(result)

                    # Now execute the query with new indexes
                    try:
                        result = CypherTool(db=db).call({"query": query})
                        return [
                            {
                                "query_result": result,
                                "indexes_created": created_indexes,
                                "message": (
                                    "Indexes created and query executed " "successfully"
                                ),
                            }
                        ]
                    except Exception as e:
                        return [
                            {
                                "error": (
                                    f"Error executing query after creating "
                                    f"indexes: {str(e)}"
                                ),
                                "indexes_created": created_indexes,
                            }
                        ]
                else:  # User clicked "No"
                    logger.info("User declined index creation")
                    # Execute without creating indexes
                    try:
                        result = CypherTool(db=db).call({"query": query})
                        return [
                            {
                                "query_result": result,
                                "status": "success",
                                "message": (
                                    "Query executed without index "
                                    "optimization (user declined)"
                                ),
                                "missing_indexes": index_queries,
                            }
                        ]
                    except Exception as e:
                        return [
                            {
                                "error": f"Error executing query: {str(e)}",
                                "status": "error",
                            }
                        ]

            elif elicit_result.action == "decline":
                logger.info("User declined the elicitation")
                # Execute without creating indexes
                try:
                    result = CypherTool(db=db).call({"query": query})
                    return [
                        {
                            "query_result": result,
                            "status": "success",
                            "message": ("Query executed without index optimization"),
                            "missing_indexes": index_queries,
                        }
                    ]
                except Exception as e:
                    return [
                        {
                            "error": f"Error executing query: {str(e)}",
                            "status": "error",
                        }
                    ]

            else:  # cancel
                logger.info("User cancelled operation")
                return [
                    {
                        "status": "cancelled",
                        "message": "Query execution cancelled by user",
                    }
                ]

        except Exception as e:
            logger.error("Elicitation failed: %s", str(e))
            # Fall back to executing without indexes
            result = CypherTool(db=db).call({"query": query})
            return [
                {
                    "query_result": result,
                    "warning": (
                        "Elicitation not supported, executed without " "optimization"
                    ),
                    "missing_indexes": index_queries,
                }
            ]

    # Step 5: Execute query if all indexes exist or none are needed
    logger.info("All recommended indexes exist or none needed, executing query")

    try:
        result = CypherTool(db=db).call({"query": query})
        return [
            {
                "query_result": result,
                "status": "success",
                "message": ("Query executed successfully with optimal indexes"),
            }
        ]
    except Exception as e:
        return [
            {
                "error": f"Error executing query: {str(e)}",
                "status": "error",
            }
        ]


def create_index_helper(label: str, property: str, index_type: str) -> Dict[str, Any]:
    """Helper function to create an index."""
    try:
        if index_type == "label":
            # Label-only index (no property)
            query = f"CREATE INDEX ON :{label}"
        elif index_type == "vector":
            # Vector index creation - use default dimension and capacity
            query = (
                f"CREATE VECTOR INDEX ON :{label}({property}) "
                "WITH CONFIG {'dimension': 384, 'capacity': 10000}"
            )
        elif index_type == "text":
            # Text index creation
            query = f"CREATE TEXT INDEX ON :{label}({property})"
        else:
            # Label+property index
            query = f"CREATE INDEX ON :{label}({property})"

        logger.info("Creating index with query: %s", query)
        db.query(query)

        return {
            "label": label,
            "property": property,
            "type": index_type,
            "status": "created",
            "query": query,
        }

    except Exception as e:
        logger.error("Error creating index: %s", str(e))
        return {
            "label": label,
            "property": property,
            "type": index_type,
            "status": "error",
            "error": str(e),
        }


@mcp.tool()
def create_index(
    label: str, property: str, index_type: str = "label+property"
) -> Dict[str, Any]:
    """
    Create an index in Memgraph.

    Args:
        label: The node label for the index
        property: The property name for the index
        index_type: Type of index - 'vector', 'text', or
                    'label+property' (default)

    Returns:
        Dictionary with creation status
    """
    logger.info("Create index called: %s.%s (%s)", label, property, index_type)
    return create_index_helper(label, property, index_type)


@mcp.tool()
def get_index_info() -> List[Dict[str, Any]]:
    """
    Get information about all indexes in Memgraph.

    Returns:
        List of index information including type, label, and properties
    """
    logger.info("Getting index information")
    try:
        indexes = ShowIndexInfoTool(db=db).call({})
        return indexes
    except Exception as e:
        return [{"error": f"Error getting index info: {str(e)}"}]


@mcp.tool()
def get_schema_info() -> List[Dict[str, Any]]:
    """
    Get schema information from Memgraph including labels and
    relationship types.

    Returns:
        Schema information
    """
    logger.info("Getting schema information")
    try:
        schema = ShowSchemaInfoTool(db=db).call({})
        return schema
    except Exception as e:
        return [{"error": f"Error getting schema info: {str(e)}"}]


@mcp.tool()
async def analyze_query(query: str, ctx: Context) -> Dict[str, Any]:
    """
    Analyze a Cypher query to understand index requirements without
    executing it.

    This tool uses sampling to check what indexes would be beneficial
    for the query.

    Args:
        query: The Cypher query to analyze
        ctx: FastMCP context for sampling

    Returns:
        Analysis results including recommended indexes
    """
    logger.info("Analyzing query: %s", query)

    try:
        # Get query analysis using sampling
        analysis = await analyze_query_with_sampling(query, ctx)

        # Get current indexes
        current_indexes = get_current_indexes()

        # Check which recommended indexes are missing
        missing_indexes = []
        existing_indexes = []

        for rec_idx in analysis["recommended_indexes"]:
            label = rec_idx["label"]
            prop = rec_idx["property"]
            idx_type = rec_idx["type"]

            if idx_type == "label":
                exists = current_indexes.has_label_index(label)
            elif idx_type == "vector":
                exists = current_indexes.has_vector_index(label, prop)
            elif idx_type == "text":
                exists = current_indexes.has_text_index(label, prop)
            else:
                exists = current_indexes.has_label_property_index(label, prop)

            if exists:
                existing_indexes.append(rec_idx)
            else:
                missing_indexes.append(rec_idx)

        return {
            "query": query,
            "labels": analysis["labels"],
            "properties": analysis["properties"],
            "uses_vector_search": analysis["uses_vector_search"],
            "uses_text_search": analysis["uses_text_search"],
            "recommended_indexes": analysis["recommended_indexes"],
            "existing_indexes": existing_indexes,
            "missing_indexes": missing_indexes,
            "optimization_potential": len(missing_indexes) > 0,
        }

    except Exception as e:
        return {"error": f"Error analyzing query: {str(e)}"}


logger.info("🔬 Memgraph Experimental MCP server initialized")
logger.info(
    "Available tools: query_tool, create_index, get_index_info, "
    "get_schema_info, analyze_query"
)
