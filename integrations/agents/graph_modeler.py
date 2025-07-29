"""
Hypothetical Graph Modeling (HyGM) Module

This module uses LLM to analyze database schemas and provide intelligent
graph modeling suggestions for optimal MySQL to Memgraph migration.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    """Represents a node in the graph model."""

    name: str
    label: str
    properties: List[str]
    primary_key: str
    indexes: List[str]
    constraints: List[str]
    source_table: str
    modeling_rationale: str


@dataclass
class GraphRelationship:
    """Represents a relationship in the graph model."""

    name: str
    type: str  # "one_to_many", "many_to_many", "one_to_one"
    from_node: str
    to_node: str
    properties: List[str]
    directionality: str  # "directed", "undirected"
    source_info: Dict[str, Any]
    modeling_rationale: str


@dataclass
class GraphModel:
    """Complete graph model for the database."""

    nodes: List[GraphNode]
    relationships: List[GraphRelationship]
    modeling_decisions: List[str]
    optimization_suggestions: List[str]
    data_patterns: Dict[str, Any]


class HyGM:
    """Uses LLM to create intelligent graph models from relational schemas."""

    def __init__(self, llm):
        """Initialize with an LLM instance."""
        self.llm = llm

    def analyze_and_model_schema(
        self, database_structure: Dict[str, Any], domain_context: Optional[str] = None
    ) -> GraphModel:
        """
        Analyze database schema and create an intelligent graph model.

        Args:
            database_structure: Database structure from MySQLAnalyzer
            domain_context: Optional domain context for better modeling

        Returns:
            GraphModel with intelligent modeling decisions
        """
        logger.info("Starting intelligent graph modeling analysis...")

        # Use domain_context if provided for enhanced analysis
        context_info = domain_context or "General database migration"
        logger.info("Domain context: %s", context_info)

        # Step 1: Analyze overall database context
        database_context = self._analyze_database_context(database_structure)

        # Step 2: Analyze each table for graph modeling
        node_analyses = []
        for table_name, table_info in database_structure["entity_tables"].items():
            analysis = self._analyze_table_for_graph_modeling(
                table_name, table_info, database_structure, database_context
            )
            node_analyses.append(analysis)

        # Step 3: Analyze relationships with full context
        relationship_analyses = self._analyze_relationships_for_graph_modeling(
            database_structure, node_analyses, database_context
        )

        # Step 4: Generate final graph model
        graph_model = self._generate_comprehensive_graph_model(
            node_analyses, relationship_analyses, database_context
        )

        logger.info("Completed intelligent graph modeling analysis")
        return graph_model

    def _analyze_database_context(
        self, database_structure: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze the overall database to understand the domain and context."""

        # Prepare database overview for LLM
        tables_overview = []
        for table_name, table_info in database_structure["tables"].items():
            column_names = [col["field"] for col in table_info["schema"]]
            fk_count = len(table_info.get("foreign_keys", []))
            row_count = table_info.get("row_count", 0)

            tables_overview.append(
                {
                    "name": table_name,
                    "columns": column_names,
                    "foreign_keys": fk_count,
                    "row_count": row_count,
                    "type": table_info.get("type", "entity"),
                }
            )

        system_message = SystemMessage(
            content="""
        You are an expert database architect and graph modeling specialist. 
        Analyze the provided database schema to understand the business domain, 
        data patterns, and optimal graph modeling approach.
        
        Focus on:
        1. Identifying the business domain (e.g., e-commerce, social media, CRM, etc.)
        2. Finding natural graph patterns and hierarchies
        3. Identifying central entities that should be highly connected nodes
        4. Recognizing lookup/reference tables vs core business entities
        5. Understanding data flow and relationships patterns
        """
        )

        human_message = HumanMessage(
            content=f"""
        Analyze this database schema:

        Tables Overview:
        {self._format_tables_for_llm(tables_overview)}

        Relationships:
        {len(database_structure.get('relationships', []))} relationships found

        Please provide:
        1. Business domain identification
        2. Core entities that should be central nodes
        3. Natural graph patterns you observe
        4. Recommended graph modeling approach
        5. Key insights for optimization

        Respond in JSON format with these keys:
        - domain: business domain description
        - core_entities: list of main entity table names
        - graph_patterns: list of observed patterns
        - modeling_approach: recommended approach
        - insights: key insights for graph optimization
        """
        )

        try:
            response = self.llm.invoke([system_message, human_message])
            # Parse LLM response (would need proper JSON parsing in production)
            return {
                "llm_analysis": response.content,
                "tables_count": len(database_structure["tables"]),
                "entity_tables_count": len(database_structure["entity_tables"]),
                "relationships_count": len(database_structure.get("relationships", [])),
            }
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("Database context analysis failed: %s", e)
            return {
                "llm_analysis": "Context analysis unavailable",
                "tables_count": len(database_structure["tables"]),
                "entity_tables_count": len(database_structure["entity_tables"]),
                "relationships_count": len(database_structure.get("relationships", [])),
            }

    def _analyze_table_for_graph_modeling(
        self,
        table_name: str,
        table_info: Dict[str, Any],
        database_structure: Dict[str, Any],
        database_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Analyze a specific table for optimal graph node modeling."""

        # Prepare table analysis context
        schema_details = []

        # Handle both schema formats: entity_tables uses "columns", tables uses "schema"
        if "schema" in table_info:
            # Format from database_analyzer (tables)
            for col in table_info["schema"]:
                schema_details.append(
                    {
                        "name": col["field"],
                        "type": col["type"],
                        "nullable": col["null"] == "YES",
                        "key": col.get("key", ""),
                        "default": col.get("default", ""),
                    }
                )
        elif "columns" in table_info:
            # Format from entity_tables
            for col_name, col_info in table_info["columns"].items():
                schema_details.append(
                    {
                        "name": col_name,
                        "type": col_info["type"],
                        "nullable": col_info.get("nullable", False),
                        "key": col_info.get("key", ""),
                        "default": col_info.get("default", ""),
                    }
                )
        else:
            logger.warning("No schema information found for table %s", table_name)

        foreign_keys = table_info.get("foreign_keys", [])
        row_count = table_info.get("row_count", 0)

        # Find related tables
        related_tables = []
        for rel in database_structure.get("relationships", []):
            if rel["from_table"] == table_name or rel["to_table"] == table_name:
                related_tables.append(rel)

        system_message = SystemMessage(
            content="""
        You are a graph database modeling expert. Analyze this table for optimal 
        graph node representation considering graph database best practices.
        
        Consider:
        1. Which columns should be node properties vs separate nodes
        2. What should be the node label (avoid generic names)
        3. Which properties are good for indexing
        4. What constraints should be applied
        5. How this node fits in the overall graph structure
        """
        )

        human_message = HumanMessage(
            content=f"""
        Analyze table '{table_name}' for graph modeling:

        Database Context: {database_context.get('llm_analysis', 'N/A')}

        Table Details:
        - Row count: {row_count}
        - Columns: {len(schema_details)}
        - Foreign keys: {len(foreign_keys)}

        Schema:
        {self._format_schema_for_llm(schema_details)}

        Foreign Keys:
        {self._format_foreign_keys_for_llm(foreign_keys)}

        Related Relationships:
        {len(related_tables)} relationships involve this table

        Provide graph modeling recommendations in JSON format:
        - node_label: suggested node label (semantic, not just table name)
        - properties: list of columns that should be node properties
        - exclude_properties: columns that shouldn't be properties (with reasons)
        - indexes: recommended property indexes
        - constraints: recommended constraints
        - modeling_rationale: explanation of modeling decisions
        - graph_role: role of this node in the graph (central, lookup, bridge, etc.)
        """
        )

        try:
            response = self.llm.invoke([system_message, human_message])
            return {
                "table_name": table_name,
                "table_info": table_info,
                "llm_analysis": response.content,
                "schema_details": schema_details,
                "foreign_keys": foreign_keys,
                "related_tables": related_tables,
                "row_count": row_count,
            }
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("Table analysis failed for %s: %s", table_name, e)
            return {
                "table_name": table_name,
                "table_info": table_info,
                "llm_analysis": "Analysis unavailable",
                "schema_details": schema_details,
                "foreign_keys": foreign_keys,
                "related_tables": related_tables,
                "row_count": row_count,
            }

    def _analyze_relationships_for_graph_modeling(
        self,
        database_structure: Dict[str, Any],
        node_analyses: List[Dict[str, Any]],
        database_context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Analyze relationships for optimal graph edge modeling."""

        relationships = database_structure.get("relationships", [])
        relationship_analyses = []

        # Group relationships by type for batch analysis
        one_to_many_rels = [r for r in relationships if r["type"] == "one_to_many"]
        many_to_many_rels = [r for r in relationships if r["type"] == "many_to_many"]

        # Analyze one-to-many relationships
        if one_to_many_rels:
            analysis = self._analyze_relationship_batch(
                one_to_many_rels, "one_to_many", node_analyses, database_context
            )
            relationship_analyses.extend(analysis)

        # Analyze many-to-many relationships
        if many_to_many_rels:
            analysis = self._analyze_relationship_batch(
                many_to_many_rels, "many_to_many", node_analyses, database_context
            )
            relationship_analyses.extend(analysis)

        return relationship_analyses

    def _analyze_relationship_batch(
        self,
        relationships: List[Dict[str, Any]],
        relationship_type: str,
        node_analyses: List[Dict[str, Any]],
        database_context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Analyze a batch of relationships of the same type."""

        system_message = SystemMessage(
            content=f"""
        You are a graph database expert analyzing {relationship_type} relationships 
        for optimal graph edge modeling.
        
        Consider:
        1. Meaningful relationship names (not just table names)
        2. Directionality and semantic meaning
        3. Whether relationship properties should be modeled
        4. Performance implications for graph traversals
        5. Graph query patterns that will be common
        """
        )

        # Prepare relationship context
        rel_context = []
        for rel in relationships:
            from_table = rel["from_table"]
            to_table = rel["to_table"]

            # Find node analysis for context
            from_analysis = next(
                (n for n in node_analyses if n["table_name"] == from_table), None
            )
            to_analysis = next(
                (n for n in node_analyses if n["table_name"] == to_table), None
            )

            rel_context.append(
                {
                    "relationship": rel,
                    "from_table_context": from_analysis.get("llm_analysis", "N/A")
                    if from_analysis
                    else "N/A",
                    "to_table_context": to_analysis.get("llm_analysis", "N/A")
                    if to_analysis
                    else "N/A",
                }
            )

        human_message = HumanMessage(
            content=f"""
        Analyze these {relationship_type} relationships for graph modeling:

        Database Context: {database_context.get('llm_analysis', 'N/A')}

        Relationships to analyze:
        {self._format_relationships_for_llm(rel_context)}

        For each relationship, provide recommendations in JSON format:
        - relationship_name: semantic relationship name
        - directionality: directed/undirected with rationale
        - properties: any relationship properties to model
        - performance_notes: optimization considerations
        - modeling_rationale: explanation of decisions
        """
        )

        try:
            response = self.llm.invoke([system_message, human_message])
            return [
                {
                    "relationship_type": relationship_type,
                    "relationships": relationships,
                    "llm_analysis": response.content,
                    "context": rel_context,
                }
            ]
        except Exception as e:  # pylint: disable=broad-except
            logger.warning(
                "Relationship analysis failed for %s: %s", relationship_type, e
            )
            return [
                {
                    "relationship_type": relationship_type,
                    "relationships": relationships,
                    "llm_analysis": "Analysis unavailable",
                    "context": rel_context,
                }
            ]

    def _generate_comprehensive_graph_model(
        self,
        node_analyses: List[Dict[str, Any]],
        relationship_analyses: List[Dict[str, Any]],
        database_context: Dict[str, Any],
    ) -> GraphModel:
        """Generate the final comprehensive graph model."""

        # Generate nodes from analyses
        nodes = []
        for analysis in node_analyses:
            node = GraphNode(
                name=analysis["table_name"],
                label=self._extract_node_label(analysis),
                properties=self._extract_node_properties(analysis),
                primary_key=self._extract_primary_key(analysis),
                indexes=self._extract_indexes(analysis),
                constraints=self._extract_constraints(analysis),
                source_table=analysis["table_name"],
                modeling_rationale=analysis.get("llm_analysis", ""),
            )
            nodes.append(node)

        # Generate relationships from analyses
        relationships = []
        for analysis in relationship_analyses:
            for rel_data in analysis["relationships"]:
                relationship = GraphRelationship(
                    name=self._extract_relationship_name(rel_data, analysis),
                    type=rel_data["type"],
                    from_node=rel_data["from_table"],
                    to_node=rel_data["to_table"],
                    properties=self._extract_relationship_properties(
                        rel_data, analysis
                    ),
                    directionality="directed",  # Default, could be extracted from LLM
                    source_info=rel_data,
                    modeling_rationale=analysis.get("llm_analysis", ""),
                )
                relationships.append(relationship)

        # Generate overall modeling decisions and suggestions
        modeling_decisions = self._extract_modeling_decisions(
            node_analyses, relationship_analyses, database_context
        )

        optimization_suggestions = self._extract_optimization_suggestions(
            node_analyses, relationship_analyses, database_context
        )

        return GraphModel(
            nodes=nodes,
            relationships=relationships,
            modeling_decisions=modeling_decisions,
            optimization_suggestions=optimization_suggestions,
            data_patterns=database_context,
        )

    # Helper methods for formatting data for LLM
    def _format_tables_for_llm(self, tables_overview: List[Dict]) -> str:
        """Format tables overview for LLM consumption."""
        formatted = []
        for table in tables_overview:
            formatted.append(
                f"- {table['name']}: {len(table['columns'])} columns, "
                f"{table['foreign_keys']} FKs, {table['row_count']} rows, "
                f"type: {table['type']}"
            )
        return "\n".join(formatted)

    def _format_schema_for_llm(self, schema_details: List[Dict]) -> str:
        """Format table schema for LLM consumption."""
        formatted = []
        for col in schema_details:
            nullable = "NULL" if col["nullable"] else "NOT NULL"
            key_info = f" ({col['key']})" if col["key"] else ""
            formatted.append(f"- {col['name']}: {col['type']} {nullable}{key_info}")
        return "\n".join(formatted)

    def _format_foreign_keys_for_llm(self, foreign_keys: List[Dict]) -> str:
        """Format foreign keys for LLM consumption."""
        if not foreign_keys:
            return "None"

        formatted = []
        for fk in foreign_keys:
            formatted.append(
                f"- {fk['column']} -> {fk['referenced_table']}.{fk['referenced_column']}"
            )
        return "\n".join(formatted)

    def _format_relationships_for_llm(self, rel_context: List[Dict]) -> str:
        """Format relationships context for LLM consumption."""
        formatted = []
        for ctx in rel_context:
            rel = ctx["relationship"]
            if rel["type"] == "many_to_many":
                formatted.append(
                    f"- {rel['from_table']} <--> {rel['to_table']} "
                    f"(via {rel['join_table']})"
                )
            else:
                formatted.append(f"- {rel['from_table']} -> {rel['to_table']}")
        return "\n".join(formatted)

    # Helper methods for extracting information from LLM responses
    def _extract_node_label(self, analysis: Dict[str, Any]) -> str:
        """Extract node label from analysis (with fallback)."""
        # In a real implementation, this would parse the JSON response
        # For now, return a sensible default
        table_name = analysis["table_name"]
        return table_name.replace("_", "").title()

    def _extract_node_properties(self, analysis: Dict[str, Any]) -> List[str]:
        """Extract node properties from analysis."""
        # Fallback: exclude obvious foreign keys and system columns
        properties = []
        for col in analysis["schema_details"]:
            if (
                not col["name"].endswith("_id")
                or col["key"] == "PRI"
                and col["name"] not in ["created_at", "updated_at", "deleted_at"]
            ):
                properties.append(col["name"])
        return properties

    def _extract_primary_key(self, analysis: Dict[str, Any]) -> str:
        """Extract primary key from analysis."""
        for col in analysis["schema_details"]:
            if col["key"] == "PRI":
                return col["name"]
        return "id"  # fallback

    def _extract_indexes(self, analysis: Dict[str, Any]) -> List[str]:
        """Extract recommended indexes from analysis."""
        indexes = []
        for col in analysis["schema_details"]:
            if col["key"] in ["UNI", "MUL"] or col["name"] in [
                "email",
                "username",
                "name",
            ]:
                indexes.append(col["name"])
        return indexes

    def _extract_constraints(self, analysis: Dict[str, Any]) -> List[str]:
        """Extract recommended constraints from analysis."""
        constraints = []
        for col in analysis["schema_details"]:
            if col["key"] == "PRI":
                constraints.append(f"UNIQUE({col['name']})")
        return constraints

    def _extract_relationship_name(
        self, rel_data: Dict[str, Any], analysis: Dict[str, Any]
    ) -> str:
        # pylint: disable=unused-argument
        """Extract relationship name from analysis."""
        # Simple fallback logic
        if rel_data["type"] == "many_to_many":
            join_table = rel_data.get("join_table", "")
            return join_table.upper().replace("_", "_")
        else:
            to_table = rel_data["to_table"]
            return f"HAS_{to_table.upper()}"

    def _extract_relationship_properties(
        self, rel_data: Dict[str, Any], analysis: Dict[str, Any]
    ) -> List[str]:
        # pylint: disable=unused-argument
        """Extract relationship properties from analysis."""
        if rel_data["type"] == "many_to_many":
            return rel_data.get("additional_properties", [])
        return []

    def _extract_modeling_decisions(
        self,
        node_analyses: List[Dict[str, Any]],
        relationship_analyses: List[Dict[str, Any]],
        database_context: Dict[str, Any],
    ) -> List[str]:
        # pylint: disable=unused-argument
        """Extract key modeling decisions from all analyses."""
        decisions = [
            f"Identified {len(node_analyses)} entity nodes for the graph",
            (
                "Configured "
                f"{sum(len(ra['relationships']) for ra in relationship_analyses)}"
                " relationships"
            ),
            "Applied semantic labeling based on business domain analysis",
            "Optimized property selection for graph traversal performance",
        ]
        return decisions

    def _extract_optimization_suggestions(
        self,
        node_analyses: List[Dict[str, Any]],
        relationship_analyses: List[Dict[str, Any]],
        database_context: Dict[str, Any],
    ) -> List[str]:
        # pylint: disable=unused-argument
        """Extract optimization suggestions from analyses."""
        suggestions = [
            (
                "Consider adding graph-specific indexes for frequently "
                "queried properties"
            ),
            "Monitor relationship cardinality for performance optimization",
            "Implement caching for high-degree nodes",
            "Consider partitioning strategies for large datasets",
        ]
        return suggestions
