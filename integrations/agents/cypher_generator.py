"""
Cypher query generator for converting MySQL schema to Memgraph.
"""

from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class CypherGenerator:
    """Generates Cypher queries for Memgraph based on MySQL schema."""

    def __init__(self):
        """Initialize the Cypher generator."""
        self.type_mapping = {
            "int": "INTEGER",
            "bigint": "INTEGER",
            "smallint": "INTEGER",
            "tinyint": "INTEGER",
            "varchar": "STRING",
            "char": "STRING",
            "text": "STRING",
            "longtext": "STRING",
            "mediumtext": "STRING",
            "decimal": "FLOAT",
            "float": "FLOAT",
            "double": "FLOAT",
            "datetime": "DATETIME",
            "timestamp": "DATETIME",
            "date": "DATE",
            "time": "TIME",
            "enum": "STRING",
            "set": "STRING",
            "blob": "STRING",
            "json": "STRING",
        }

    def mysql_to_cypher_type(self, mysql_type: str) -> str:
        """Convert MySQL data type to Cypher/Memgraph type."""
        # Extract base type (remove size specifications)
        base_type = mysql_type.split("(")[0].lower()
        return self.type_mapping.get(base_type, "STRING")

    def generate_node_creation_query(
        self, table_name: str, schema: List[Dict[str, Any]]
    ) -> str:
        """Generate Cypher query to create nodes for a table."""
        # Determine primary key
        primary_keys = [col["field"] for col in schema if col["key"] == "PRI"]

        if not primary_keys:
            # If no primary key, use first field as identifier
            id_field = schema[0]["field"] if schema else "id"
        else:
            id_field = primary_keys[0]

        # Create property definitions
        properties = []
        for col in schema:
            if col["field"] != id_field:  # Skip the ID field in properties
                cypher_type = self.mysql_to_cypher_type(col["type"])
                properties.append(f"{col['field']}: {cypher_type}")

        # Generate the query
        label = self._table_name_to_label(table_name)
        query = f"""
        // Create {label} nodes
        UNWIND $data AS row
        CREATE (n:{label} {{
            {id_field}: row.{id_field}"""

        if properties:
            query += ",\n            " + ",\n            ".join(
                f"{prop.split(':')[0]}: row.{prop.split(':')[0]}" for prop in properties
            )

        query += "\n        })"

        return query.strip()

    def generate_relationship_query(
        self, from_table: str, from_column: str, to_table: str, to_column: str
    ) -> str:
        """Generate Cypher query to create relationships."""
        from_label = self._table_name_to_label(from_table)
        to_label = self._table_name_to_label(to_table)
        rel_type = self._generate_relationship_type(from_table, to_table)

        query = f"""
        // Create {rel_type} relationships from {from_label} to {to_label}
        MATCH (from:{from_label})
        MATCH (to:{to_label})
        WHERE from.{from_column} = to.{to_column}
        CREATE (from)-[:{rel_type}]->(to)
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
                index_type = "UNIQUE" if col["key"] in ["PRI", "UNI"] else ""
                query = f"CREATE {index_type} INDEX ON :{label}({col['field']})"
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
        for table_name, table_info in structure["tables"].items():
            constraint_queries = self.generate_constraint_queries(
                table_name, table_info["schema"]
            )
            queries.extend(constraint_queries)
        queries.append("")

        # 2. Create indexes
        queries.append("// Step 2: Create indexes")
        for table_name, table_info in structure["tables"].items():
            index_queries = self.generate_index_queries(
                table_name, table_info["schema"]
            )
            queries.extend(index_queries)
        queries.append("")

        # 3. Create nodes
        queries.append("// Step 3: Create nodes")
        for table_name, table_info in structure["tables"].items():
            node_query = self.generate_node_creation_query(
                table_name, table_info["schema"]
            )
            queries.append(node_query)
            queries.append("")

        # 4. Create relationships
        queries.append("// Step 4: Create relationships")
        for rel in structure["relationships"]:
            rel_query = self.generate_relationship_query(
                rel["from_table"], rel["from_column"], rel["to_table"], rel["to_column"]
            )
            queries.append(rel_query)
            queries.append("")

        return queries

    def _table_name_to_label(self, table_name: str) -> str:
        """Convert table name to Cypher label."""
        # Convert to PascalCase
        return "".join(word.capitalize() for word in table_name.split("_"))

    def _generate_relationship_type(self, from_table: str, to_table: str) -> str:
        """Generate relationship type name."""
        # Create a meaningful relationship name
        from_label = self._table_name_to_label(from_table)
        to_label = self._table_name_to_label(to_table)

        # Common relationship patterns
        if "customer" in from_table.lower() and "order" in to_table.lower():
            return "PLACED"
        elif "order" in from_table.lower() and "item" in to_table.lower():
            return "CONTAINS"
        elif "film" in from_table.lower() and "actor" in to_table.lower():
            return "FEATURES"
        elif "actor" in from_table.lower() and "film" in to_table.lower():
            return "ACTED_IN"
        elif "store" in from_table.lower():
            return "BELONGS_TO"
        elif "address" in to_table.lower():
            return "LOCATED_AT"
        elif "category" in to_table.lower():
            return "BELONGS_TO_CATEGORY"
        elif "language" in to_table.lower():
            return "IN_LANGUAGE"
        else:
            return f"RELATED_TO_{to_label.upper()}"

    def prepare_data_for_cypher(
        self, data: List[Dict[str, Any]], schema: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Prepare data for Cypher ingestion by handling null values
        and type conversions."""
        prepared_data = []

        for row in data:
            prepared_row = {}
            for col in schema:
                field_name = col["field"]
                value = row.get(field_name)

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
                    cypher_type = self.mysql_to_cypher_type(col["type"])
                    if cypher_type in ["DATETIME", "DATE", "TIME"]:
                        # Convert datetime objects to strings
                        value = str(value) if value else None

                prepared_row[field_name] = value

            prepared_data.append(prepared_row)

        return prepared_data
