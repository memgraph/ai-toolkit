"""
Memgraph database adapter for schema validation.

This adapter provides connection and query capabilities for Memgraph
to support post-migration validation.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    # Imported directly (rather than referenced as `neo4j.Driver`) so its
    # type is precise even though the `neo4j` name itself is inferred as
    # `<module 'neo4j'> | None` below to model the optional-import fallback.
    from neo4j import Driver
    from typing_extensions import LiteralString

try:
    import neo4j
except ImportError:
    neo4j = None

try:
    # mgclient (the native Memgraph client) is a genuinely optional driver:
    # unlike neo4j/psycopg2/mysql-connector below, it is never declared as a
    # dependency of this package at all, so ty has no stub to resolve it
    # against -- there is no fix short of adding a real dependency, which is
    # out of scope for a types-only pass.
    import mgclient  # ty: ignore[unresolved-import]
except ImportError:
    mgclient = None


logger = logging.getLogger(__name__)


@dataclass
class MemgraphConnectionConfig:
    """Configuration for Memgraph connection."""

    host: str = "localhost"
    port: int = 7687
    username: str = ""
    password: str = ""
    use_ssl: bool = False


class MemgraphAdapter:
    """
    Adapter for connecting to and querying Memgraph database.

    Supports both mgclient and neo4j driver connections.
    """

    def __init__(self, config: MemgraphConnectionConfig):
        """
        Initialize Memgraph adapter.

        Args:
            config: Connection configuration
        """
        self.config = config
        # mgclient's connection type is unresolvable (see the import above),
        # so this is genuinely Any; neo4j is a real dependency (declared in
        # pyproject.toml) so its Driver type is precise.
        self.connection: Any = None
        self.driver: Driver | None = None
        self._connection_type: str | None = None

    def connect(self) -> bool:
        """
        Establish connection to Memgraph.

        Returns:
            True if connection successful, False otherwise
        """
        # Try mgclient first (native Memgraph client)
        if mgclient and self._try_mgclient_connection():
            self._connection_type = "mgclient"
            logger.info("Connected to Memgraph using mgclient")
            return True

        # Fallback to neo4j driver (compatible with Memgraph)
        if neo4j and self._try_neo4j_connection():
            self._connection_type = "neo4j"
            logger.info("Connected to Memgraph using neo4j driver")
            return True

        logger.error("Failed to connect to Memgraph with available drivers")
        return False

    def _try_mgclient_connection(self) -> bool:
        """Try to connect using mgclient."""
        try:
            # Only reached when connect() has already checked `mgclient` is
            # truthy, an invariant this private method's own signature can't
            # express since the check lives in the caller.
            mg = cast("Any", mgclient)
            connection = mg.connect(
                host=self.config.host,
                port=self.config.port,
                username=self.config.username,
                password=self.config.password,
                sslmode=mg.SSLMode.REQUIRE if self.config.use_ssl else mg.SSLMode.DISABLE,
            )
            self.connection = connection
            # Test the connection
            cursor = connection.cursor()
            cursor.execute("RETURN 1")
            cursor.fetchall()
            return True
        except Exception as e:
            logger.debug(f"mgclient connection failed: {e}")
            return False

    def _try_neo4j_connection(self) -> bool:
        """Try to connect using neo4j driver."""
        try:
            # Only reached when connect() has already checked `neo4j` is
            # truthy; see the comment on _try_mgclient_connection above.
            nx = cast("Any", neo4j)
            uri = f"{'bolt+s' if self.config.use_ssl else 'bolt'}://{self.config.host}:{self.config.port}"
            driver = nx.GraphDatabase.driver(
                uri,
                auth=(self.config.username, self.config.password) if self.config.username else None,
            )
            self.driver = driver
            # Test the connection
            with driver.session() as session:
                session.run("RETURN 1")
            return True
        except Exception as e:
            logger.debug(f"neo4j driver connection failed: {e}")
            return False

    def execute_query(self, query: str) -> list[tuple]:
        """
        Execute a query and return results.

        Args:
            query: Cypher query to execute

        Returns:
            List of result tuples
        """
        if not self.connection and not self.driver:
            raise RuntimeError("Not connected to Memgraph")

        if self._connection_type == "mgclient":
            return self._execute_mgclient_query(query)
        elif self._connection_type == "neo4j":
            return self._execute_neo4j_query(query)
        else:
            raise RuntimeError("Unknown connection type")

    def _execute_mgclient_query(self, query: str) -> list[tuple]:
        """Execute query using mgclient."""
        cursor = self.connection.cursor()
        cursor.execute(query)
        return cursor.fetchall()

    def _execute_neo4j_query(self, query: str) -> list[tuple]:
        """Execute query using neo4j driver."""
        # execute_query() (the only caller) already checked _connection_type
        # == "neo4j" before dispatching here, which is only ever set right
        # after a successful self.driver assignment in _try_neo4j_connection.
        driver = cast("Driver", self.driver)
        with driver.session() as session:
            # Cypher text is built dynamically at runtime, so it can never be
            # a `LiteralString` in the sense neo4j's typed API wants; that
            # constraint exists to steer callers away from string-building
            # untrusted input, which isn't how queries are built here.
            result = session.run(cast("LiteralString", query))
            return [tuple(record.values()) for record in result]

    def get_schema_info(self) -> list[tuple]:
        """
        Get schema information from Memgraph.

        Returns:
            Schema information as list of tuples
        """
        try:
            # Try SHOW SCHEMA INFO first (newer Memgraph versions)
            return self.execute_query("SHOW SCHEMA INFO")
        except Exception:
            # Fallback to alternative schema queries
            logger.warning("SHOW SCHEMA INFO not available, using alternative schema detection")
            return self._get_schema_info_alternative()

    def _get_schema_info_alternative(self) -> list[tuple]:
        """
        Alternative method to get schema info for older Memgraph versions.

        Returns:
            Schema information as list of tuples
        """
        schema_info = []

        # Get node labels and their properties
        try:
            node_labels_result = self.execute_query("CALL db.labels()")
            for label_row in node_labels_result:
                label = label_row[0]

                # Get properties for this label
                props_query = f"MATCH (n:{label}) RETURN DISTINCT keys(n) LIMIT 1"
                props_result = self.execute_query(props_query)
                properties = props_result[0][0] if props_result and props_result[0] else []

                # Format as schema info tuple
                schema_info.append(("node", label, {prop: "Unknown" for prop in properties}))
        except Exception as e:
            logger.warning(f"Failed to get node labels: {e}")

        # Get relationship types and their properties
        try:
            rel_types_result = self.execute_query("CALL db.relationshipTypes()")
            for rel_row in rel_types_result:
                rel_type = rel_row[0]

                # Get properties for this relationship type
                props_query = f"MATCH ()-[r:{rel_type}]-() RETURN DISTINCT keys(r) LIMIT 1"
                props_result = self.execute_query(props_query)
                properties = props_result[0][0] if props_result and props_result[0] else []

                # Format as schema info tuple
                schema_info.append(("relationship", rel_type, {prop: "Unknown" for prop in properties}))
        except Exception as e:
            logger.warning(f"Failed to get relationship types: {e}")

        return schema_info

    def get_indexes(self) -> list[dict[str, Any]]:
        """
        Get index information from Memgraph.

        Returns:
            List of index information dictionaries
        """
        try:
            indexes_result = self.execute_query("SHOW INDEX INFO")
            indexes = []

            for index_row in indexes_result:
                # Parse index information based on Memgraph's SHOW INDEX INFO format
                if len(index_row) >= 3:
                    index_info = {
                        "name": index_row[0],
                        "type": index_row[1],
                        "properties": index_row[2] if len(index_row) > 2 else [],
                    }
                    indexes.append(index_info)

            return indexes
        except Exception as e:
            logger.warning(f"Failed to get index information: {e}")
            return []

    def get_constraints(self) -> list[dict[str, Any]]:
        """
        Get constraint information from Memgraph.

        Returns:
            List of constraint information dictionaries
        """
        try:
            constraints_result = self.execute_query("SHOW CONSTRAINT INFO")
            constraints = []

            for constraint_row in constraints_result:
                # Parse constraint information based on Memgraph's format
                if len(constraint_row) >= 3:
                    constraint_info = {
                        "name": constraint_row[0],
                        "type": constraint_row[1],
                        "properties": constraint_row[2] if len(constraint_row) > 2 else [],
                    }
                    constraints.append(constraint_info)

            return constraints
        except Exception as e:
            logger.warning(f"Failed to get constraint information: {e}")
            return []

    def close(self):
        """Close the database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None
        if self.driver:
            self.driver.close()
            self.driver = None

        logger.info("Memgraph connection closed")

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def create_memgraph_connection(
    host: str = "localhost",
    port: int = 7687,
    username: str = "",
    password: str = "",
    use_ssl: bool = False,
) -> MemgraphAdapter:
    """
    Create a Memgraph connection adapter.

    Args:
        host: Memgraph host
        port: Memgraph port
        username: Username for authentication
        password: Password for authentication
        use_ssl: Whether to use SSL connection

    Returns:
        MemgraphAdapter instance
    """
    config = MemgraphConnectionConfig(host=host, port=port, username=username, password=password, use_ssl=use_ssl)

    return MemgraphAdapter(config)
