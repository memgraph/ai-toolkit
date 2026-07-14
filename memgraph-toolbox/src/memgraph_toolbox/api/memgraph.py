import os
from collections.abc import Mapping
from typing import Any

from neo4j import AsyncGraphDatabase, GraphDatabase

from ..utils.serialization import serialize_record_data

MEMGRAPH_ENV_DEFAULTS = {
    "MEMGRAPH_URL": "bolt://localhost:7687",
    "MEMGRAPH_USER": "",
    "MEMGRAPH_PASSWORD": "",
    "MEMGRAPH_DATABASE": "memgraph",
}
MEMGRAPH_ENV_KEYS = tuple(MEMGRAPH_ENV_DEFAULTS)


def memgraph_env(
    *,
    url: str | None = None,
    username: str | None = None,
    password: str | None = None,
    database: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return canonical Memgraph connection environment values.

    Explicit values win, then environment variables, then Memgraph defaults.
    The returned keys are suitable for passing through command hooks or other
    subprocess boundaries.
    """
    env = os.environ if environ is None else environ
    return {
        "MEMGRAPH_URL": url if url is not None else env.get("MEMGRAPH_URL", MEMGRAPH_ENV_DEFAULTS["MEMGRAPH_URL"]),
        "MEMGRAPH_USER": (
            username if username is not None else env.get("MEMGRAPH_USER", MEMGRAPH_ENV_DEFAULTS["MEMGRAPH_USER"])
        ),
        "MEMGRAPH_PASSWORD": (
            password
            if password is not None
            else env.get("MEMGRAPH_PASSWORD", MEMGRAPH_ENV_DEFAULTS["MEMGRAPH_PASSWORD"])
        ),
        "MEMGRAPH_DATABASE": (
            database
            if database is not None
            else env.get("MEMGRAPH_DATABASE", MEMGRAPH_ENV_DEFAULTS["MEMGRAPH_DATABASE"])
        ),
    }


class Memgraph:
    """
    Base Memgraph client for interacting with Memgraph database.
    """

    DEFAULT_USER_AGENT = "memgraph-toolbox"

    def __init__(
        self,
        url: str = None,
        username: str = None,
        password: str = None,
        database: str = None,
        driver_config: dict | None = None,
        user_agent: str | None = None,
    ):
        """
        Initialize Memgraph client with connection parameters.

        Connection details can be provided directly or via environment variables:
        - MEMGRAPH_URL (default: "bolt://localhost:7687")
        - MEMGRAPH_USER (default: "")
        - MEMGRAPH_PASSWORD (default: "")
        - MEMGRAPH_DATABASE (default: "memgraph")

        Args:
            url: The Memgraph connection URL
            username: Username for authentication
            password: Password for authentication
            database: The database name to connect to (default: "memgraph")
            driver_config: Additional Neo4j driver configuration
            user_agent: Client name sent to the server (e.g. "mcp-memgraph", "langchain-memgraph", "sql2graph", etc.)
        """

        # Load from the shared Memgraph environment contract with fallbacks.
        env = memgraph_env(url=url, username=username, password=password, database=database)
        url = env["MEMGRAPH_URL"]
        username = env["MEMGRAPH_USER"]
        password = env["MEMGRAPH_PASSWORD"]

        config = dict(driver_config or {})
        config.setdefault("user_agent", user_agent or self.DEFAULT_USER_AGENT)

        self.driver = GraphDatabase.driver(url, auth=(username, password), **config)

        self.database = env["MEMGRAPH_DATABASE"]

        try:
            import neo4j
        except ImportError as e:
            raise ImportError(
                "Could not import neo4j python package. Please install it with `pip install neo4j`."
            ) from e
        try:
            self.driver.verify_connectivity()
        except neo4j.exceptions.ServiceUnavailable as e:
            raise ValueError(f"Could not connect to Memgraph database. Please ensure the URL '{url}' is correct") from e
        except neo4j.exceptions.AuthError as e:
            raise ValueError(
                f"Could not connect to Memgraph database. Authentication failed for user '{username}'"
            ) from e

    def query(self, query: str, params: dict = None) -> list[dict[str, Any]]:
        """
        Execute a Cypher query and return results as a list of dictionaries.

        Args:
            query: The Cypher query to execute

        Returns:
            List of dictionaries containing query results
        """
        from neo4j.exceptions import Neo4jError

        if params is None:
            params = {}
        try:
            data, _, _ = self.driver.execute_query(
                query,
                parameters_=params,
                database_=self.database,
            )
            json_data = [serialize_record_data(r.data()) for r in data]
            return json_data
        except Neo4jError as e:
            if not (
                (
                    (  # isCallInTransactionError
                        e.code == "Neo.DatabaseError.Statement.ExecutionFailed"
                        or e.code == "Neo.DatabaseError.Transaction.TransactionStartFailed"
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
                or (e.code == "Memgraph.ClientError.MemgraphError.MemgraphError" and "SchemaInfo disabled" in e.message)
            ):
                raise

        # fallback to allow implicit transactions
        with self.driver.session(database=self.database) as session:
            data = session.run(query, params)
            json_data = [serialize_record_data(r.data()) for r in data]
            return json_data

    def close(self) -> None:
        """
        Close the database connection.
        """
        self.driver.close()


class AsyncMemgraph:
    """
    Async Memgraph client mirroring :class:`Memgraph`.

    Uses ``neo4j.AsyncGraphDatabase`` so callers running inside an event loop can
    reuse the same connection contract (env vars, defaults and ``query``
    semantics) as the sync client. Connectivity is intentionally NOT verified in
    ``__init__`` (the async driver cannot be awaited there); call
    :meth:`verify_connectivity` explicitly when a live check is needed.

    The raw ``driver`` and ``database`` are exposed as public attributes so
    callers that need to open their own async sessions can do so.
    """

    DEFAULT_USER_AGENT = "memgraph-toolbox"

    def __init__(
        self,
        url: str = None,
        username: str = None,
        password: str = None,
        database: str = None,
        driver_config: dict | None = None,
        user_agent: str | None = None,
    ):
        """
        Initialize the async Memgraph client with connection parameters.

        Connection details can be provided directly or via environment variables:
        - MEMGRAPH_URL (default: "bolt://localhost:7687")
        - MEMGRAPH_USER (default: "")
        - MEMGRAPH_PASSWORD (default: "")
        - MEMGRAPH_DATABASE (default: "memgraph")

        Args:
            url: The Memgraph connection URL
            username: Username for authentication
            password: Password for authentication
            database: The database name to connect to (default: "memgraph")
            driver_config: Additional Neo4j driver configuration
            user_agent: Client name sent to the server (e.g. "mcp-memgraph", "langchain-memgraph", "sql2graph", etc.)
        """

        # Resolve config through the same single source of truth as the sync client.
        env = memgraph_env(url=url, username=username, password=password, database=database)
        url = env["MEMGRAPH_URL"]
        username = env["MEMGRAPH_USER"]
        password = env["MEMGRAPH_PASSWORD"]

        config = dict(driver_config or {})
        config.setdefault("user_agent", user_agent or self.DEFAULT_USER_AGENT)

        self.driver = AsyncGraphDatabase.driver(url, auth=(username, password), **config)
        self.database = env["MEMGRAPH_DATABASE"]

    async def verify_connectivity(self) -> None:
        """
        Verify that the server is reachable and credentials are valid.

        Call this explicitly after construction when a live check is desired.
        """
        await self.driver.verify_connectivity()

    async def query(self, query: str, params: dict = None) -> list[dict[str, Any]]:
        """
        Execute a Cypher query and return results as a list of dictionaries.

        Mirrors :meth:`Memgraph.query`, including the fallback to an implicit
        transaction for statements that cannot run inside a managed transaction.

        Args:
            query: The Cypher query to execute
            params: Optional query parameters

        Returns:
            List of dictionaries containing query results
        """
        from neo4j.exceptions import Neo4jError

        if params is None:
            params = {}
        try:
            data, _, _ = await self.driver.execute_query(
                query,
                parameters_=params,
                database_=self.database,
            )
            json_data = [serialize_record_data(r.data()) for r in data]
            return json_data
        except Neo4jError as e:
            if not (
                (
                    (  # isCallInTransactionError
                        e.code == "Neo.DatabaseError.Statement.ExecutionFailed"
                        or e.code == "Neo.DatabaseError.Transaction.TransactionStartFailed"
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
                or (e.code == "Memgraph.ClientError.MemgraphError.MemgraphError" and "SchemaInfo disabled" in e.message)
            ):
                raise

        # fallback to allow implicit transactions
        async with self.driver.session(database=self.database) as session:
            result = await session.run(query, params)
            json_data = [serialize_record_data(r.data()) async for r in result]
            return json_data

    async def close(self) -> None:
        """
        Close the database connection.
        """
        await self.driver.close()
