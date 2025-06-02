from typing import Any, Dict, List, Optional
import os
from neo4j import GraphDatabase


class Memgraph:
    """
    Base Memgraph client for interacting with Memgraph database.
    """

    def __init__(
        self,
        url: str = None,
        username: str = None,
        password: str = None,
        driver_config: Optional[Dict] = None,
    ):
        """
        Initialize Memgraph client with connection parameters.

        Connection details can be provided directly or via environment variables:
        - MEMGRAPH_URL (default: "bolt://localhost:7687")
        - MEMGRAPH_USER (default: "")
        - MEMGRAPH_PASSWORD (default: "")

        Args:
            url: The Memgraph connection URL
            username: Username for authentication
            password: Password for authentication
            driver_config: Additional Neo4j driver configuration
        """

        # Load from environment variables with fallbacks
        url = url or os.environ.get("MEMGRAPH_URL", "bolt://localhost:7687")
        username = username or os.environ.get("MEMGRAPH_USER", "")
        password = password or os.environ.get("MEMGRAPH_PASSWORD", "")

        self.driver = GraphDatabase.driver(
            url, auth=(username, password), **(driver_config or {})
        )
        try:
            import neo4j
        except ImportError:
            raise ImportError(
                "Could not import neo4j python package. "
                "Please install it with `pip install neo4j`."
            )
        try:
            self.driver.verify_connectivity()
        except neo4j.exceptions.ServiceUnavailable:
            raise ValueError(
                "Could not connect to Memgraph database. "
                f"Please ensure the URL '{url}' is correct"
            )
        except neo4j.exceptions.AuthError:
            raise ValueError(
                "Could not connect to Memgraph database. "
                f"Authentication failed for user '{username}'"
            )

    def query(self, query: str, params: dict = {}) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query and return results as a list of dictionaries.

        Args:
            query: The Cypher query to execute

        Returns:
            List of dictionaries containing query results
        """
        from neo4j.exceptions import Neo4jError

        try:
            data, _, _ = self.driver.execute_query(
                query,
                parameters_=params,
            )
            json_data = [r.data() for r in data]
            return json_data
        except Neo4jError as e:
            if not (
                (
                    (  # isCallInTransactionError
                        e.code == "Neo.DatabaseError.Statement.ExecutionFailed"
                        or e.code
                        == "Neo.DatabaseError.Transaction.TransactionStartFailed"
                    )
                    and "in an implicit transaction" in e.message
                )
                or (  # isPeriodicCommitError
                    e.code == "Neo.ClientError.Statement.SemanticError"
                    and (
                        "in an open transaction is not possible" in e.message
                        or "tried to execute in an explicit transaction" in e.message
                    )
                )
                or (
                    e.code == "Memgraph.ClientError.MemgraphError.MemgraphError"
                    and ("in multicommand transactions" in e.message)
                )
                or (
                    e.code == "Memgraph.ClientError.MemgraphError.MemgraphError"
                    and "SchemaInfo disabled" in e.message
                )
            ):
                raise

        # fallback to allow implicit transactions
        with self.driver.session() as session:
            data = session.run(query, params)
            json_data = [r.data() for r in data]
            return json_data

    def close(self) -> None:
        """
        Close the database connection.
        """
        self.driver.close()
