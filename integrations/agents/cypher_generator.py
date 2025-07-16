"""
Cypher query generator for converting MySQL schema to Memgraph.
"""

from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class CypherGenerator:
    """Generates Cypher queries for Memgraph based on MySQL schema."""

    def __init__(self, relationship_naming_strategy: str = "table_based"):
        """Initialize the Cypher generator.

        Args:
            relationship_naming_strategy: Strategy for naming relationships.
                - "table_based": Use table names directly (default)
                - "llm": Use LLM to generate meaningful names (requires LLM)
        """
        self.relationship_naming_strategy = relationship_naming_strategy
        self.llm = None  # Will be set if using LLM strategy

        self.type_mapping = {
            "int": "INTEGER",
            "bigint": "INTEGER",
            "smallint": "INTEGER",
            "tinyint": "INTEGER",
            "mediumint": "INTEGER",
            "varchar": "STRING",
            "char": "STRING",
            "text": "STRING",
            "longtext": "STRING",
            "mediumtext": "STRING",
            "tinytext": "STRING",
            "decimal": "FLOAT",
            "numeric": "FLOAT",
            "float": "FLOAT",
            "double": "FLOAT",
            "real": "FLOAT",
            "datetime": "DATETIME",
            "timestamp": "DATETIME",
            "date": "DATE",
            "time": "TIME",
            "year": "INTEGER",
            "enum": "STRING",
            "set": "STRING",
            "blob": "STRING",
            "tinyblob": "STRING",
            "mediumblob": "STRING",
            "longblob": "STRING",
            "binary": "STRING",
            "varbinary": "STRING",
            "json": "STRING",
            "geometry": "STRING",
            "point": "STRING",
            "linestring": "STRING",
            "polygon": "STRING",
            "bit": "INTEGER",
        }

    def set_llm(self, llm):
        """Set LLM for relationship naming strategy."""
        self.llm = llm

    def mysql_to_cypher_type(self, mysql_type: str) -> str:
        """Convert MySQL data type to Cypher/Memgraph type."""
        # Extract base type (remove size specifications)
        base_type = mysql_type.split("(")[0].lower()
        return self.type_mapping.get(base_type, "STRING")

    def generate_node_creation_query(
        self,
        table_name: str,
        schema: List[Dict[str, Any]],
        foreign_keys: List[Dict[str, str]] = None,
    ) -> str:
        """Generate Cypher query to create nodes for a table."""
        if foreign_keys is None:
            foreign_keys = []

        # Determine primary key
        primary_keys = [col["field"] for col in schema if col["key"] == "PRI"]

        if not primary_keys:
            # If no primary key, use first field as identifier
            id_field = schema[0]["field"] if schema else "id"
        else:
            id_field = primary_keys[0]

        # Get foreign key column names to exclude them from properties
        fk_column_names = {fk["column"] for fk in foreign_keys}

        # Create property definitions (exclude FK columns and ID field)
        properties = []
        for col in schema:
            if col["field"] != id_field and col["field"] not in fk_column_names:
                safe_field_name = self._escape_reserved_keyword(col["field"])
                cypher_type = self.mysql_to_cypher_type(col["type"])
                properties.append(f"{safe_field_name}: {cypher_type}")

        # Generate the query
        label = self._table_name_to_label(table_name)
        safe_id_field = self._escape_reserved_keyword(id_field)

        query = f"""
        // Create {label} nodes
        UNWIND $data AS row
        CREATE (n:{label} {{
            {safe_id_field}: row.{safe_id_field}"""

        if properties:
            property_assignments = []
            for col in schema:
                if col["field"] != id_field and col["field"] not in fk_column_names:
                    safe_field_name = self._escape_reserved_keyword(col["field"])
                    property_assignments.append(
                        f"{safe_field_name}: row.{safe_field_name}"
                    )

            query += ",\n            " + ",\n            ".join(property_assignments)

        query += "\n        })"

        return query.strip()

    def generate_relationship_query(self, relationship: Dict[str, Any]) -> str:
        """Generate Cypher query to create relationships."""
        if relationship["type"] == "many_to_many":
            return self._generate_many_to_many_query(relationship)
        else:
            return self._generate_one_to_many_query(relationship)

    def _generate_one_to_many_query(self, relationship: Dict[str, Any]) -> str:
        """Generate one-to-many relationship query."""
        from_table = relationship["from_table"]
        from_column = relationship["from_column"]
        to_table = relationship["to_table"]
        to_column = relationship["to_column"]

        from_label = self._table_name_to_label(from_table)
        to_label = self._table_name_to_label(to_table)
        rel_type = self._generate_relationship_type(from_table, to_table)

        # Escape column names if they are reserved keywords
        safe_from_column = self._escape_reserved_keyword(from_column)
        safe_to_column = self._escape_reserved_keyword(to_column)

        query = f"""
        // Create {rel_type} relationships from {from_label} to {to_label}
        MATCH (from:{from_label})
        MATCH (to:{to_label})
        WHERE from.{safe_from_column} = to.{safe_to_column}
        CREATE (from)-[:{rel_type}]->(to)
        """

        return query.strip()

    def _generate_many_to_many_query(self, relationship: Dict[str, Any]) -> str:
        """Generate many-to-many relationship query from join table."""
        join_table = relationship["join_table"]
        from_table = relationship["from_table"]
        to_table = relationship["to_table"]
        join_from_column = relationship["join_from_column"]
        join_to_column = relationship["join_to_column"]
        from_column = relationship["from_column"]
        to_column = relationship["to_column"]
        additional_properties = relationship.get("additional_properties", [])

        from_label = self._table_name_to_label(from_table)
        to_label = self._table_name_to_label(to_table)
        rel_type = self._generate_relationship_type(from_table, to_table, join_table)

        # Escape column names if they are reserved keywords
        safe_from_column = self._escape_reserved_keyword(from_column)
        safe_to_column = self._escape_reserved_keyword(to_column)
        safe_join_from_column = self._escape_reserved_keyword(join_from_column)
        safe_join_to_column = self._escape_reserved_keyword(join_to_column)

        # Build relationship properties
        rel_properties = ""
        if additional_properties:
            prop_assignments = []
            for prop in additional_properties:
                safe_prop = self._escape_reserved_keyword(prop)
                prop_assignments.append(f"{safe_prop}: row.{safe_prop}")
            rel_properties = " {" + ", ".join(prop_assignments) + "}"

        query = f"""
        // Create {rel_type} relationships from {from_label} to {to_label}
        // via {join_table} join table
        UNWIND $data AS row
        MATCH (from:{from_label} {{{safe_from_column}: row.{safe_join_from_column}}})
        MATCH (to:{to_label} {{{safe_to_column}: row.{safe_join_to_column}}})
        CREATE (from)-[:{rel_type}{rel_properties}]->(to)
        """

        return query.strip()

    def generate_index_queries(
        self, table_name: str, schema: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate index creation queries."""
        queries = []
        label = self._table_name_to_label(table_name)

        for col in schema:
            if col["key"] in ["PRI", "UNI", "MUL"]:
                query = f"CREATE INDEX ON :{label}({col['field']})"
                queries.append(query.strip())

        return queries

    def generate_constraint_queries(
        self, table_name: str, schema: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate constraint creation queries."""
        queries = []
        label = self._table_name_to_label(table_name)

        # Primary key constraints
        primary_keys = [col["field"] for col in schema if col["key"] == "PRI"]
        for pk in primary_keys:
            query = f"CREATE CONSTRAINT ON (n:{label}) ASSERT n.{pk} IS UNIQUE"
            queries.append(query)

        # Unique constraints
        unique_keys = [col["field"] for col in schema if col["key"] == "UNI"]
        for uk in unique_keys:
            query = f"CREATE CONSTRAINT ON (n:{label}) ASSERT n.{uk} IS UNIQUE"
            queries.append(query)

        return queries

    def generate_full_migration_script(self, structure: Dict[str, Any]) -> List[str]:
        """Generate complete migration script."""
        queries = []

        # Add header comment
        queries.append("// MySQL to Memgraph Migration Script")
        queries.append("// Generated automatically")
        queries.append("")

        # 1. Create constraints first
        queries.append("// Step 1: Create constraints")
        for table_name, table_info in structure["entity_tables"].items():
            constraint_queries = self.generate_constraint_queries(
                table_name, table_info["schema"]
            )
            queries.extend(constraint_queries)
        queries.append("")

        # 2. Create indexes
        queries.append("// Step 2: Create indexes")
        for table_name, table_info in structure["entity_tables"].items():
            index_queries = self.generate_index_queries(
                table_name, table_info["schema"]
            )
            queries.extend(index_queries)
        queries.append("")

        # 3. Create nodes (only for entity tables, not join tables)
        queries.append("// Step 3: Create nodes")
        for table_name, table_info in structure["entity_tables"].items():
            node_query = self.generate_node_creation_query(
                table_name, table_info["schema"], table_info["foreign_keys"]
            )
            queries.append(node_query)
            queries.append("")

        # 4. Create relationships
        queries.append("// Step 4: Create relationships")
        for rel in structure["relationships"]:
            rel_query = self.generate_relationship_query(rel)
            queries.append(rel_query)
            queries.append("")

        return queries

    def _table_name_to_label(self, table_name: str) -> str:
        """Convert table name to Cypher label."""
        # Convert to PascalCase
        return "".join(word.capitalize() for word in table_name.split("_"))

    def _generate_relationship_type(
        self, from_table: str, to_table: str, join_table: str = None
    ) -> str:
        """Generate relationship type name."""
        if self.relationship_naming_strategy == "llm" and self.llm:
            return self._generate_relationship_type_with_llm(
                from_table, to_table, join_table
            )
        else:
            # Table-based naming strategy (default)
            if join_table:
                return self._table_name_to_label(join_table).upper()
            else:
                return f"HAS_{self._table_name_to_label(to_table).upper()}"

    def _generate_relationship_type_with_llm(
        self, from_table: str, to_table: str, join_table: str = None
    ) -> str:
        """Generate relationship type using LLM."""
        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            if join_table:
                prompt = f"""
                Given a many-to-many relationship between tables '{from_table}'
                and '{to_table}' via join table '{join_table}', suggest a
                meaningful relationship name in UPPER_CASE format.

                Examples:
                - film_actor -> ACTED_IN
                - customer_rental -> RENTED
                - user_role -> HAS_ROLE

                Return only the relationship name, nothing else.
                """
            else:
                prompt = f"""
                Given a one-to-many relationship from table '{from_table}'
                to table '{to_table}', suggest a meaningful relationship name
                in UPPER_CASE format.

                Examples:
                - customer -> order: PLACED
                - order -> order_item: CONTAINS
                - film -> language: IN_LANGUAGE

                Return only the relationship name, nothing else.
                """

            messages = [
                SystemMessage(content="You are a database modeling expert."),
                HumanMessage(content=prompt),
            ]

            response = self.llm.invoke(messages)
            relationship_name = response.content.strip().upper()

            # Validate the response
            if relationship_name and relationship_name.replace("_", "").isalpha():
                return relationship_name
            else:
                # Fallback to table-based naming
                if join_table:
                    return self._table_name_to_label(join_table).upper()
                else:
                    return f"HAS_{self._table_name_to_label(to_table).upper()}"

        except Exception as e:
            logger.warning(f"LLM relationship naming failed: {e}")
            # Fallback to table-based naming
            if join_table:
                return self._table_name_to_label(join_table).upper()
            else:
                return f"HAS_{self._table_name_to_label(to_table).upper()}"

    def prepare_data_for_cypher(
        self,
        data: List[Dict[str, Any]],
        schema: List[Dict[str, Any]],
        foreign_keys: List[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """Prepare data for Cypher ingestion by handling null values
        and type conversions."""
        if foreign_keys is None:
            foreign_keys = []

        prepared_data = []
        fk_column_names = {fk["column"] for fk in foreign_keys}

        for row_idx, row in enumerate(data):
            prepared_row = {}
            try:
                for col in schema:
                    field_name = col["field"]
                    value = row.get(field_name)

                    # Skip foreign key columns and non-existent columns
                    if field_name in fk_column_names or field_name not in row:
                        continue

                    # Handle reserved keywords by escaping them
                    safe_field_name = self._escape_reserved_keyword(field_name)

                    # Handle null values
                    if value is None:
                        if col["null"] == "NO":
                            # Set default value for non-nullable fields
                            cypher_type = self.mysql_to_cypher_type(col["type"])
                            if cypher_type == "INTEGER":
                                value = 0
                            elif cypher_type == "FLOAT":
                                value = 0.0
                            elif cypher_type == "STRING":
                                value = ""
                            else:
                                value = None
                        else:
                            value = None

                    # Convert types if needed
                    if value is not None:
                        try:
                            value = self._convert_value_for_cypher(value, col["type"])
                        except Exception as e:
                            logger.warning(
                                f"Failed to convert value {value} for field {field_name}: {e}"
                            )
                            # Try to convert to string as fallback
                            value = str(value) if value is not None else None

                    prepared_row[safe_field_name] = value

                prepared_data.append(prepared_row)

            except Exception as e:
                logger.error(f"Error preparing row {row_idx}: {e}")
                # Skip this row rather than failing the entire migration
                continue

        return prepared_data

    def prepare_join_table_data_for_cypher(
        self, data: List[Dict[str, Any]], schema: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Prepare join table data for relationship creation."""
        prepared_data = []

        for row in data:
            prepared_row = {}
            for col in schema:
                field_name = col["field"]
                value = row.get(field_name)

                # Handle reserved keywords by escaping them
                safe_field_name = self._escape_reserved_keyword(field_name)

                # Convert types if needed
                if value is not None:
                    value = self._convert_value_for_cypher(value, col["type"])

                prepared_row[safe_field_name] = value

            prepared_data.append(prepared_row)

        return prepared_data

    def _convert_value_for_cypher(self, value: Any, mysql_type: str) -> Any:
        """Convert a MySQL value to Cypher-compatible type."""
        import decimal
        from datetime import datetime, date, time

        # Handle decimal types
        if isinstance(value, decimal.Decimal):
            return float(value)

        # Handle datetime types
        if isinstance(value, (datetime, date, time)):
            return str(value)

        # Handle boolean types (MySQL uses tinyint(1))
        if mysql_type.lower().startswith("tinyint(1)"):
            return bool(value)

        # Handle MySQL SET types (convert to comma-separated string)
        if isinstance(value, set):
            return ",".join(sorted(str(item) for item in value))

        # Handle MySQL ENUM types if they come as sets
        if mysql_type.lower().startswith("set") and isinstance(value, str):
            # Already a string, keep as is
            return value

        # Handle binary data
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8")
            except UnicodeDecodeError:
                # For binary data that can't be decoded, convert to hex string
                return value.hex()

        # Handle large integers that might exceed JavaScript safe integer range
        if isinstance(value, int) and abs(value) > 2**53 - 1:
            return str(value)

        # Handle None/NULL values
        if value is None:
            return None

        # Handle any other types by converting to string
        if not isinstance(value, (str, int, float, bool)):
            return str(value)

        return value

    def _escape_reserved_keyword(self, field_name: str) -> str:
        """Escape Cypher reserved keywords and problematic field names."""
        # Cypher reserved keywords that need to be escaped
        reserved_keywords = {
            "code",
            "data",
            "type",
            "name",
            "value",
            "id",
            "count",
            "size",
            "match",
            "where",
            "return",
            "create",
            "delete",
            "set",
            "remove",
            "merge",
            "order",
            "by",
            "limit",
            "skip",
            "with",
            "union",
            "all",
            "distinct",
            "optional",
            "foreach",
            "case",
            "when",
            "then",
            "else",
            "end",
            "and",
            "or",
            "xor",
            "not",
            "in",
            "starts",
            "ends",
            "contains",
            "is",
            "null",
            "unique",
            "index",
            "on",
            "drop",
            "constraint",
            "assert",
            "scan",
            "using",
            "join",
            "start",
            "node",
            "relationship",
            "rel",
            "shortestpath",
            "allshortestpaths",
            "extract",
            "filter",
            "reduce",
            "any",
            "none",
            "single",
            "true",
            "false",
            "load",
            "csv",
            "from",
            "as",
            "into",
            "to",
            "explain",
            "profile",
            "call",
            "yield",
            "periodic",
            "commit",
            "transaction",
            "begin",
            "rollback",
            "show",
            "create",
            "drop",
            "exists",
            "labels",
            "keys",
            "nodes",
            "relationships",
            "procedures",
            "functions",
            "database",
            "databases",
            "default",
            "user",
            "users",
            "role",
            "roles",
            "privilege",
            "privileges",
            "grant",
            "deny",
            "revoke",
            "catalog",
            "schema",
            "schemas",
        }

        # Always escape field names with spaces or special characters
        if (
            " " in field_name
            or any(char in field_name for char in ["-", ".", "@", "#", "$", "%"])
            or field_name.lower() in reserved_keywords
        ):
            return f"`{field_name}`"
        return field_name
