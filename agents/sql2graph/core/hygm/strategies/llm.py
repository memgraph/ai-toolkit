"""
LLM-powered modeling strategy for Hypothetical Graph Modeling (HyGM).

This strategy uses AI/LLM models to create sophisticated graph models.
Supports multiple providers: OpenAI, Anthropic, Gemini via LangChain.
"""

import logging
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from langchain_core.language_models import BaseChatModel

if TYPE_CHECKING:
    from core.hygm.models.graph_models import GraphModel

try:
    from .base import BaseModelingStrategy
except ImportError:
    from core.hygm.strategies.base import BaseModelingStrategy

logger = logging.getLogger(__name__)


class LLMStrategy(BaseModelingStrategy):
    """
    LLM-powered graph modeling strategy using AI for intelligent mapping.

    Uses LangChain's BaseChatModel interface to support multiple providers.
    """

    def __init__(
        self,
        llm_client: Optional[BaseChatModel] = None,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.1,
    ):
        """
        Initialize LLM strategy.

        Args:
            llm_client: LangChain chat model (ChatOpenAI/ChatAnthropic/ChatGoogleGenerativeAI)
            model_name: Model to use for graph generation
            temperature: Temperature for generation (lower=more deterministic)
        """
        self.llm_client = llm_client
        self.model_name = model_name
        self.temperature = temperature
        self._database_structure = {}
        self._current_llm_model = None

    def get_strategy_name(self) -> str:
        """Return the name of this strategy."""
        return "llm"

    def create_model(
        self,
        database_structure: Dict[str, Any],
        domain_context: Optional[str] = None,
        user_operation_context: Optional[str] = None,
    ) -> "GraphModel":
        """
        Create a sophisticated graph model using LLM analysis.

        Args:
            database_structure: Database schema and structure
            domain_context: Optional domain context for better modeling
            user_operation_context: Context about user operations to preserve

        Returns:
            GraphModel: AI-generated graph model

        Raises:
            ValueError: If no LLM client is provided
            Exception: If LLM model creation fails
        """
        logger.info("Creating LLM-powered graph model...")

        # Store database structure for relationship mapping
        self._database_structure = database_structure

        if not self.llm_client:
            raise ValueError(
                "No LLM client provided for LLM-powered modeling. "
                "Please configure OpenAI API key or use deterministic strategy."
            )

        return self._create_llm_model(
            database_structure, domain_context, user_operation_context
        )

    def _create_llm_model(
        self,
        database_structure: Dict[str, Any],
        domain_context: Optional[str] = None,
        user_operation_context: Optional[str] = None,
    ) -> "GraphModel":
        """Create model using LLM structured output."""
        logger.info("Using LLM to generate graph model...")

        try:
            # Import the structured output model
            from core.hygm.models.llm_models import LLMGraphModel

            # Prepare the prompt
            prompt = self._build_modeling_prompt(
                database_structure, domain_context, user_operation_context
            )

            # Call LLM with unified client interface
            if not self.llm_client:
                raise ValueError("No LLM client available")

            # Create system message for graph modeling
            system_message = (
                "You are an expert database architect specializing "
                "in converting relational schemas to graph models. "
                "Analyze the provided database structure and create "
                "an optimal graph model that preserves relationships "
                "and enables efficient querying."
            )

            # Generate the structured output using LangChain's with_structured_output
            structured_llm = self.llm_client.with_structured_output(LLMGraphModel)
            llm_model = structured_llm.invoke(
                [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ]
            )

            logger.info(
                "LLM generated %d nodes and %d relationships",
                len(llm_model.nodes),
                len(llm_model.relationships),
            )

            # Extract and convert LLM response to internal graph model
            return self._convert_llm_to_graph_model(llm_model)

        except ImportError as e:
            error_msg = f"Missing LLM models module: {e}"
            logger.error(error_msg)
            raise ImportError(error_msg) from e
        except Exception as e:
            error_msg = f"LLM model creation failed: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def _build_modeling_prompt(
        self,
        database_structure: Dict[str, Any],
        domain_context: Optional[str] = None,
        user_operation_context: Optional[str] = None,
    ) -> str:
        """Build the prompt for LLM graph modeling."""

        prompt_parts = [
            "Convert this relational database schema to an optimal graph model.",
            "Analyze the database structure and create nodes and relationships that:",
            "",
            "Database Structure:",
            str(database_structure),
            "",
            "Requirements:",
            "- Create semantic node labels (not just table names)",
            "- Identify meaningful relationships based on foreign keys",
            "- Include relevant properties from source tables on nodes",
            "- Set appropriate primary keys for each node",
            "- **CRITICAL: Recommend indexes for ALL relation properties**",
            "- **CRITICAL: Both relationship ends MUST have indexes**",
            "- Include unique constraints for primary keys and unique fields",
            "- Use descriptive relationship names (e.g., 'OWNS', 'BELONGS_TO')",
            "- Consider one-to-many, many-to-many, and one-to-one types",
            "- Optimize for both data integrity and query performance",
            "",
            "Index Guidelines:",
            "- Primary keys should always have indexes",
            "- Foreign key properties should always have indexes",
            "- Unique fields should have indexes",
            "- Properties in WHERE clauses should have indexes",
            "",
            "Constraint Guidelines:",
            "- Primary key properties should have unique constraints",
            "- Unique business fields should have unique constraints",
            "- Consider data integrity requirements from source database",
        ]

        # Add user operation context if provided (critical for preserving user changes)
        if user_operation_context:
            prompt_parts.extend(
                [
                    "",
                    "⚠️ CRITICAL REQUIREMENT:",
                    user_operation_context,
                    "⚠️ YOU MUST PRESERVE ALL USER CHANGES LISTED ABOVE.",
                    "DO NOT REVERT ANY USER OPERATIONS WHEN CREATING THE MODEL.",
                    "",
                ]
            )

        if domain_context:
            prompt_parts.extend(
                [
                    "",
                    f"Domain Context: {domain_context}",
                    "Use this context to create more semantically meaningful "
                    "models.",
                ]
            )

        return "\n".join(prompt_parts)

    def _convert_llm_to_graph_model(self, llm_model) -> "GraphModel":
        """Convert LLM response to internal GraphModel format."""
        # Store the LLM model for use in relationship mapping
        self._current_llm_model = llm_model

        from core.hygm.models.graph_models import (
            GraphModel,
            GraphNode,
            GraphRelationship,
            GraphProperty,
            GraphIndex,
            GraphConstraint,
        )
        from core.hygm.models.sources import (
            NodeSource,
            PropertySource,
            RelationshipSource,
            IndexSource,
            ConstraintSource,
        )

        # Convert nodes - preserve original table mapping
        nodes = []
        for llm_node in llm_model.nodes:
            source = NodeSource(
                type="table",  # Keep as table source for migration
                name=llm_node.source_table,  # Use actual source table name
                location=f"database.schema.{llm_node.source_table}",
                mapping={"labels": llm_node.labels},
            )

            properties = []
            for prop_name in llm_node.properties:
                field_path = f"{llm_node.source_table}.{prop_name}"
                prop_source = PropertySource(field=field_path)
                graph_prop = GraphProperty(key=prop_name, source=prop_source)
                properties.append(graph_prop)

            node = GraphNode(
                labels=llm_node.labels, properties=properties, source=source
            )
            nodes.append(node)

        # Convert relationships - preserve database structure mapping
        relationships = []
        for llm_rel in llm_model.relationships:
            # Find the source database relationship information
            db_rel_info = self._find_database_relationship(llm_rel)

            if db_rel_info:
                # Use actual database structure for migration mapping
                from_table = db_rel_info.get("from_table", "")
                location = f"database.schema.{from_table}"
                rel_source = RelationshipSource(
                    type=db_rel_info.get("source_type", "table"),
                    name=db_rel_info.get("constraint_name", llm_rel.name),
                    location=location,
                    mapping=db_rel_info.get("mapping", {}),
                )
            else:
                # Get source tables for the relationship nodes
                from_table = self._get_source_table_for_node(llm_rel.from_node)
                to_table = self._get_source_table_for_node(llm_rel.to_node)

                # If we can't find direct mapping, create basic table mapping
                if from_table and to_table:
                    # Get primary keys for proper mapping
                    from_pk = self._get_table_primary_key(from_table)
                    to_pk = self._get_table_primary_key(to_table)

                    mapping = {
                        "start_node": f"{from_table}.{from_pk}",
                        "end_node": f"{to_table}.{to_pk}",
                        "from_pk": from_pk,
                        "edge_type": llm_rel.name,
                        "directionality": llm_rel.directionality,
                    }
                    location = f"database.schema.{from_table}"
                else:
                    mapping = {
                        "edge_type": llm_rel.name,
                        "directionality": llm_rel.directionality,
                        # Still need start_node and end_node for migration agent
                        "start_node": "unknown.id",
                        "end_node": "unknown.id",
                        "from_pk": "id",
                    }
                    location = "ai_analysis"

                # Fallback for relationships without database mapping
                # Use "table" as a supported type for migration agent
                rel_source = RelationshipSource(
                    type="table",
                    name=llm_rel.name,
                    location=location,
                    mapping=mapping,
                )

            properties = []
            for prop_name in llm_rel.properties:
                prop_source = PropertySource(field=f"llm.{prop_name}")
                graph_prop = GraphProperty(key=prop_name, source=prop_source)
                properties.append(graph_prop)

            # Map LLM node names to actual node labels
            start_labels = self._map_node_name_to_labels(llm_rel.from_node, llm_model)
            end_labels = self._map_node_name_to_labels(llm_rel.to_node, llm_model)

            relationship = GraphRelationship(
                edge_type=llm_rel.name,
                start_node_labels=start_labels,
                end_node_labels=end_labels,
                properties=properties,
                source=rel_source,
                directionality=llm_rel.directionality,
            )
            relationships.append(relationship)

        # Convert indexes from node-level index specifications
        indexes = []
        for llm_node in llm_model.nodes:
            for index_prop in llm_node.indexes:
                index_source = IndexSource(
                    origin="llm_recommendation",
                    reason=f"Index recommended by LLM for {llm_node.name}."
                    f"{index_prop}",
                    created_by="ai_analysis",
                    index_name=None,
                    migrated_from=None,
                )

                graph_index = GraphIndex(
                    labels=llm_node.labels,
                    properties=[index_prop],
                    type="btree",  # Default index type
                    source=index_source,
                )
                indexes.append(graph_index)

        # Convert constraints from node-level constraint specifications
        constraints = []
        for llm_node in llm_model.nodes:
            for constraint_prop in llm_node.constraints:
                constraint_source = ConstraintSource(
                    origin="llm_recommendation",
                    constraint_name=f"ai_unique_constraint_{llm_node.name}_"
                    f"{constraint_prop}",
                    migrated_from="ai_analysis",
                )

                graph_constraint = GraphConstraint(
                    type="unique",  # Assume unique constraints
                    labels=llm_node.labels,
                    properties=[constraint_prop],
                    source=constraint_source,
                )
                constraints.append(graph_constraint)

        return GraphModel(
            nodes=nodes,
            edges=relationships,
            node_indexes=indexes,
            node_constraints=constraints,
        )

    def _find_database_relationship(self, llm_rel) -> Dict[str, Any]:
        """
        Find the original database relationship information for an LLM
        relationship.

        This method matches LLM-generated relationships back to the original
        database structure to preserve technical migration details.
        """
        if not hasattr(self, "_database_structure"):
            return {}

        relationships = self._database_structure.get("relationships", [])
        entity_tables = self._database_structure.get("entity_tables", {})

        # First, try to get source table names from LLM model context
        from_table_name = self._get_source_table_for_node(llm_rel.from_node)
        to_table_name = self._get_source_table_for_node(llm_rel.to_node)

        # Try to match by source table names if available
        if from_table_name and to_table_name:
            for db_rel in relationships:
                if isinstance(db_rel, dict):
                    from_table = db_rel.get("from_table", "").lower()
                    to_table = db_rel.get("to_table", "").lower()
                    rel_type = db_rel.get("relationship_type", "one_to_many")
                else:
                    # Handle object format
                    from_table = db_rel.from_table.lower()
                    to_table = db_rel.to_table.lower()
                    rel_type = db_rel.relationship_type

                # Check if this database relationship matches the source tables
                if (
                    from_table == from_table_name.lower()
                    and to_table == to_table_name.lower()
                ):
                    return self._build_relationship_mapping(db_rel, llm_rel, rel_type)

                # Also try reverse direction
                if (
                    to_table == from_table_name.lower()
                    and from_table == to_table_name.lower()
                ):
                    return self._build_relationship_mapping(
                        db_rel, llm_rel, rel_type, reverse=True
                    )

        # Fallback: Try to match by node names and relationship semantics
        for db_rel in relationships:
            if isinstance(db_rel, dict):
                from_table = db_rel.get("from_table", "").lower()
                to_table = db_rel.get("to_table", "").lower()
                rel_type = db_rel.get("relationship_type", "one_to_many")
            else:
                # Handle object format
                from_table = db_rel.from_table.lower()
                to_table = db_rel.to_table.lower()
                rel_type = db_rel.relationship_type

            # Check if this database relationship matches the LLM relationship
            if self._tables_match_nodes(
                from_table, llm_rel.from_node
            ) and self._tables_match_nodes(to_table, llm_rel.to_node):
                return self._build_relationship_mapping(db_rel, llm_rel, rel_type)

            # Also try reverse direction for bidirectional relationships
            if self._tables_match_nodes(
                to_table, llm_rel.from_node
            ) and self._tables_match_nodes(from_table, llm_rel.to_node):
                return self._build_relationship_mapping(
                    db_rel, llm_rel, rel_type, reverse=True
                )

        # If no direct match found, try to infer from foreign keys
        return self._infer_relationship_from_foreign_keys(llm_rel, entity_tables)

    def _get_source_table_for_node(self, node_name: str) -> str:
        """Get the source table name for a given LLM node name."""
        # This should be called within _convert_llm_to_graph_model
        # where we have access to the LLM model context
        if hasattr(self, "_current_llm_model") and self._current_llm_model:
            for llm_node in self._current_llm_model.nodes:
                # Check by node name or any of the labels
                if llm_node.name.lower() == node_name.lower() or any(
                    label.lower() == node_name.lower() for label in llm_node.labels
                ):
                    return llm_node.source_table

                # Also check if node_name matches any variation of source_table
                source_table = llm_node.source_table
                if self._tables_match_nodes(source_table, node_name):
                    return source_table
        return ""

    def _build_relationship_mapping(self, db_rel, llm_rel, rel_type, reverse=False):
        """Build relationship mapping from database relationship."""
        if isinstance(db_rel, dict):
            from_table = db_rel.get("to_table" if reverse else "from_table", "")
            to_table = db_rel.get("from_table" if reverse else "to_table", "")
            from_col = db_rel.get("to_column" if reverse else "from_column", "id")
            to_col = db_rel.get("from_column" if reverse else "to_column", "id")
            join_table = db_rel.get("join_table")
            join_from_col = db_rel.get(
                "join_to_column" if reverse else "join_from_column"
            )
            join_to_col = db_rel.get(
                "join_from_column" if reverse else "join_to_column"
            )
        else:
            from_table = db_rel.to_table if reverse else db_rel.from_table
            to_table = db_rel.from_table if reverse else db_rel.to_table
            from_col = db_rel.to_column if reverse else db_rel.from_column
            to_col = db_rel.from_column if reverse else db_rel.to_column
            join_table = getattr(db_rel, "join_table", None)
            join_from_col = getattr(
                db_rel, "join_to_column" if reverse else "join_from_column", None
            )
            join_to_col = getattr(
                db_rel, "join_from_column" if reverse else "join_to_column", None
            )

        # Get primary keys for the tables from database structure
        from_table_pk = self._get_table_primary_key(from_table)

        mapping = {
            "start_node": f"{from_table}.{from_col}",
            "end_node": f"{to_table}.{to_col}",
            "edge_type": llm_rel.name,
            "from_pk": from_table_pk,  # Add primary key for migration agent
        }

        # Add many-to-many specific information if available
        if rel_type == "many_to_many" and join_table:
            mapping.update(
                {
                    "join_table": join_table,
                    "join_from_column": join_from_col,
                    "join_to_column": join_to_col,
                    "from_table": from_table,
                    "to_table": to_table,
                    "from_column": from_col,
                    "to_column": to_col,
                }
            )

        # Determine source type
        source_type = "many_to_many" if rel_type == "many_to_many" else "table"

        # Get constraint name with proper handling
        constraint_name = llm_rel.name
        if isinstance(db_rel, dict):
            constraint_name = db_rel.get("constraint_name", llm_rel.name)
        elif hasattr(db_rel, "constraint_name") and db_rel.constraint_name:
            constraint_name = db_rel.constraint_name

        return {
            "source_type": source_type,
            "constraint_name": constraint_name,
            "from_table": from_table,
            "to_table": to_table,
            "mapping": mapping,
        }

    def _infer_relationship_from_foreign_keys(self, llm_rel, entity_tables):
        """Infer relationship mapping from foreign key information."""
        # Find tables that match the relationship nodes
        from_table_name = None
        to_table_name = None

        for table_name in entity_tables.keys():
            if self._tables_match_nodes(table_name, llm_rel.from_node):
                from_table_name = table_name
            if self._tables_match_nodes(table_name, llm_rel.to_node):
                to_table_name = table_name

        if not from_table_name or not to_table_name:
            logger.warning(
                "Could not find matching tables for relationship %s: "
                "from_node=%s->%s, to_node=%s->%s",
                llm_rel.name,
                llm_rel.from_node,
                from_table_name,
                llm_rel.to_node,
                to_table_name,
            )

            # Additional debug info: show available node mappings
            if hasattr(self, "_current_llm_model") and self._current_llm_model:
                logger.debug("Available LLM nodes:")
                for node in self._current_llm_model.nodes:
                    logger.debug(
                        "  Node: %s -> Source table: %s (Labels: %s)",
                        node.name,
                        node.source_table,
                        node.labels,
                    )

            # Try additional inference for known problematic patterns
            inferred_mapping = self._infer_problematic_relationships(
                llm_rel, entity_tables
            )
            if inferred_mapping:
                return inferred_mapping

            return {}

        # Check if from_table has a foreign key to to_table
        from_table_info = entity_tables.get(from_table_name, {})
        foreign_keys = from_table_info.get("foreign_keys", [])

        for fk in foreign_keys:
            if isinstance(fk, dict):
                referenced_table = fk.get("referenced_table", "")
                fk_column = fk.get("column", "")
                referenced_column = fk.get("referenced_column", "")
            else:
                referenced_table = fk.referenced_table
                fk_column = fk.column_name
                referenced_column = fk.referenced_column

            if referenced_table.lower() == to_table_name.lower():
                # Found a matching foreign key
                from_table_pk = self._get_table_primary_key(from_table_name)
                mapping = {
                    "start_node": f"{from_table_name}.{fk_column}",
                    "end_node": f"{to_table_name}.{referenced_column}",
                    "edge_type": llm_rel.name,
                    "from_pk": from_table_pk,
                }

                return {
                    "source_type": "table",
                    "constraint_name": llm_rel.name,
                    "from_table": from_table_name,
                    "to_table": to_table_name,
                    "mapping": mapping,
                }

        # Check reverse direction
        to_table_info = entity_tables.get(to_table_name, {})
        to_foreign_keys = to_table_info.get("foreign_keys", [])

        for fk in to_foreign_keys:
            if isinstance(fk, dict):
                referenced_table = fk.get("referenced_table", "")
                fk_column = fk.get("column", "")
                referenced_column = fk.get("referenced_column", "")
            else:
                referenced_table = fk.referenced_table
                fk_column = fk.column_name
                referenced_column = fk.referenced_column

            if referenced_table.lower() == from_table_name.lower():
                # Found a reverse foreign key - reverse direction
                to_table_pk = self._get_table_primary_key(to_table_name)
                mapping = {
                    "start_node": f"{to_table_name}.{fk_column}",
                    "end_node": f"{from_table_name}.{referenced_column}",
                    "edge_type": llm_rel.name,
                    "from_pk": to_table_pk,
                }

                return {
                    "source_type": "table",
                    "constraint_name": llm_rel.name,
                    "from_table": to_table_name,
                    "to_table": from_table_name,
                    "mapping": mapping,
                }

        # Try additional inference for problematic relationships
        return self._infer_problematic_relationships(llm_rel, entity_tables)

    def _infer_problematic_relationships(self, llm_rel, entity_tables):
        """Handle relationship patterns using generic inference strategies."""

        # Try to infer relationship based on table name patterns
        from_candidates = []
        to_candidates = []

        # Look for tables that could match the relationship nodes
        for table_name in entity_tables.keys():
            if self._tables_match_nodes(table_name, llm_rel.from_node):
                from_candidates.append(table_name)
            if self._tables_match_nodes(table_name, llm_rel.to_node):
                to_candidates.append(table_name)

        # If we can't find exact matches, try pattern inference
        if not from_candidates and not to_candidates:
            return {}

        # Use the first matching candidates
        from_table = from_candidates[0] if from_candidates else None
        to_table = to_candidates[0] if to_candidates else None

        if not from_table or not to_table:
            logger.warning(
                "Could not find complete table mapping for relationship %s",
                llm_rel.name,
            )
            return {}

        # Try to infer foreign key column based on common patterns
        fk_column = self._infer_foreign_key_column(from_table, to_table, entity_tables)

        # Get primary key for from_table
        from_table_pk = self._get_table_primary_key(from_table)

        mapping = {
            "start_node": f"{from_table}.{fk_column}",
            "end_node": f"{to_table}.{fk_column}",
            "edge_type": llm_rel.name,
            "from_pk": from_table_pk,
        }

        logger.info(
            "Inferred relationship mapping for %s: %s.%s -> %s.%s",
            llm_rel.name,
            from_table,
            fk_column,
            to_table,
            fk_column,
        )

        return {
            "source_type": "table",  # Use supported type for migration agent
            "constraint_name": llm_rel.name,
            "from_table": from_table,
            "to_table": to_table,
            "mapping": mapping,
            "relationship_type": "one_to_many",  # Default assumption
        }

    def _infer_foreign_key_column(self, from_table, to_table, entity_tables):
        """Infer the most likely foreign key column name."""

        # Check actual foreign keys first
        from_table_info = entity_tables.get(from_table, {})
        foreign_keys = from_table_info.get("foreign_keys", [])

        for fk in foreign_keys:
            if isinstance(fk, dict):
                referenced_table = fk.get("referenced_table", "")
                fk_column = fk.get("column", "")
            else:
                referenced_table = fk.referenced_table
                fk_column = fk.column_name

            if referenced_table.lower() == to_table.lower():
                return fk_column

        # Get actual column names from both tables
        from_columns = []
        to_columns = []

        # Extract column names from from_table
        from_table_columns = from_table_info.get("columns", [])
        for col in from_table_columns:
            if isinstance(col, dict):
                from_columns.append(col.get("name", ""))
            else:
                from_columns.append(getattr(col, "name", str(col)))

        # Extract column names from to_table
        to_table_info = entity_tables.get(to_table, {})
        to_table_columns = to_table_info.get("columns", [])
        for col in to_table_columns:
            if isinstance(col, dict):
                to_columns.append(col.get("name", ""))
            else:
                to_columns.append(getattr(col, "name", str(col)))

        # Find common columns between the two tables
        common_columns = set(from_columns) & set(to_columns)
        if common_columns:
            # Prefer ID-like columns
            for col in ["title_id", "name_id", "id"]:
                if col in common_columns:
                    return col
            # Return the first common column
            return list(common_columns)[0]

        # Try to infer based on table names
        # For title-related relationships, use title_id
        if "title" in from_table.lower() or "title" in to_table.lower():
            if "title_id" in from_columns:
                return "title_id"
            if "title_id" in to_columns:
                return "title_id"

        # For name/person-related relationships, use name_id
        if (
            "name" in from_table.lower()
            or "name" in to_table.lower()
            or "person" in from_table.lower()
            or "person" in to_table.lower()
        ):
            if "name_id" in from_columns:
                return "name_id"
            if "name_id" in to_columns:
                return "name_id"

        # Look for any ID-like column in from_table
        for col in from_columns:
            if col.lower().endswith("_id") or col.lower() == "id":
                return col

        # Look for any ID-like column in to_table
        for col in to_columns:
            if col.lower().endswith("_id") or col.lower() == "id":
                return col

        # Ultimate fallback - use first column from from_table
        if from_columns:
            return from_columns[0]

        # Last resort
        return "id"

    def _tables_match_nodes(self, table_name: str, node_name: str) -> bool:
        """Check if a table name matches a node name (flexible matching)."""
        table_lower = table_name.lower()
        node_lower = node_name.lower()

        # Direct match
        if table_lower == node_lower:
            return True

        # Singularized match (e.g., "titles" table -> "Title" node)
        if table_lower.rstrip("s") == node_lower:
            return True

        # Pluralized match (e.g., "title" node -> "Titles" table)
        if table_lower == node_lower + "s":
            return True

        # Handle underscores to camelcase
        # (e.g., "alias_attributes" -> "AliasAttribute")
        table_camel = "".join(word.capitalize() for word in table_lower.split("_"))
        if table_camel.lower() == node_lower:
            return True

        # Handle trailing underscores (e.g., "names_" -> "Name")
        if table_lower.rstrip("_") == node_lower:
            return True
        if table_lower.rstrip("_s") == node_lower:
            return True

        # Handle reverse: node with underscores to table
        node_parts = node_lower.replace("_", " ").split()
        if len(node_parts) > 1:
            # Convert CamelCase to snake_case for comparison
            import re

            snake_case = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", node_name)
            snake_case = re.sub("([a-z0-9])([A-Z])", r"\1_\2", snake_case).lower()
            if table_lower == snake_case:
                return True

        # Handle specific patterns that are common across databases
        specific_mappings = {
            # Common pattern variations
            "genres": ["genre"],
            "ratings": ["rating"],
        }

        # Check if table has specific mapping to node
        table_base = table_lower.split("_")[-1]  # Get last part after underscore
        if table_base in specific_mappings:
            return node_lower in specific_mappings[table_base]

        # Check reverse mapping (node to table)
        for table_pattern, node_patterns in specific_mappings.items():
            if node_lower in node_patterns:
                # Check if table ends with this pattern
                if table_lower.endswith(table_pattern) or table_lower.endswith(
                    table_pattern + "s"
                ):
                    return True

        # Handle prefix patterns like "title_" + concept
        if "_" in table_lower:
            table_parts = table_lower.split("_")
            if len(table_parts) == 2:
                prefix, suffix = table_parts
                # Pattern: prefix_suffix -> Suffix (e.g., title_genres -> Genre)
                if suffix.rstrip("s") == node_lower:
                    return True

        return False

    def _map_node_name_to_labels(self, node_name: str, llm_model) -> List[str]:
        """Map LLM node name to the actual labels defined in the model."""
        for llm_node in llm_model.nodes:
            if llm_node.name.lower() == node_name.lower() or any(
                label.lower() == node_name.lower() for label in llm_node.labels
            ):
                return llm_node.labels

        # Fallback to the node name itself as label
        return [node_name]

    def _get_table_primary_key(self, table_name: str) -> str:
        """Get the primary key column name for a table from database structure."""
        if not hasattr(self, "_database_structure"):
            return f"{table_name}_id"  # Default fallback

        # Check entity tables first
        entity_tables = self._database_structure.get("entity_tables", {})
        table_info = entity_tables.get(table_name)

        if table_info:
            # Get primary keys from table info
            primary_keys = table_info.get("primary_keys", [])
            if primary_keys:
                return primary_keys[0]  # Return first primary key

            # Fallback: look in schema for primary key
            schema = table_info.get("schema", [])
            for col_info in schema:
                if isinstance(col_info, dict):
                    if col_info.get("key") == "PRI":
                        return col_info.get("field", f"{table_name}_id")
                elif hasattr(col_info, "is_primary_key") and col_info.is_primary_key:
                    return col_info.name

        # Final fallback: conventional naming
        return f"{table_name}_id"
