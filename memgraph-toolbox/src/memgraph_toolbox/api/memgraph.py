import os
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlparse

from neo4j import GraphDatabase

from ..utils.serialization import serialize_record_data

MEMGRAPH_ENV_DEFAULTS = {
    "MEMGRAPH_URL": "bolt://localhost:7687",
    "MEMGRAPH_USER": "",
    "MEMGRAPH_PASSWORD": "",
    "MEMGRAPH_DATABASE": "memgraph",
}
MEMGRAPH_ENV_KEYS = tuple(MEMGRAPH_ENV_DEFAULTS)


def _discover_main(coordinator_urls: list[str]) -> str:
    """Query coordinators to find the current MAIN data instance.

    Tries each coordinator in order. Returns a bolt:// URL to the MAIN instance.
    Raises ValueError if no coordinator is reachable or no MAIN is found.
    """
    last_error = None
    for coord_url in coordinator_urls:
        # Ensure plain bolt for coordinator queries (coordinators ignore auth).
        parsed = urlparse(coord_url)
        bolt_url = f"bolt://{parsed.hostname}:{parsed.port or 7687}"
        try:
            driver = GraphDatabase.driver(bolt_url)
            try:
                with driver.session() as session:
                    records = list(session.run("SHOW INSTANCES"))
                for record in records:
                    if record.get("role") == "main" and record.get("health") == "up":
                        bolt_server = record["bolt_server"]
                        host, _, port = bolt_server.rpartition(":")
                        # Resolve: try the reported hostname first; if it
                        # contains dots (K8s FQDN), also try the short name.
                        resolved_host = _resolve_host(host, coordinator_urls)
                        return f"bolt://{resolved_host}:{port}"
            finally:
                driver.close()
        except Exception as e:
            last_error = e
            continue
    raise ValueError(
        f"Could not discover MAIN instance from coordinators: {coordinator_urls}. Last error: {last_error}"
    )


def _resolve_host(host: str, coordinator_urls: list[str]) -> str:
    """Resolve a host reported by SHOW INSTANCES to one reachable from here.

    Memgraph reports FQDN names (e.g. memgraph-data-0.default.svc.cluster.local).
    When connecting from outside K8s, the short name (memgraph-data-0) may be
    resolvable instead. We try: FQDN → short name → match pattern from coordinators.
    """
    import socket

    # Try the full hostname first.
    try:
        socket.getaddrinfo(host, None)
        return host
    except socket.gaierror:
        pass

    # Try short hostname (first segment before first dot).
    short = host.split(".")[0]
    try:
        socket.getaddrinfo(short, None)
        return short
    except socket.gaierror:
        pass

    # Last resort: infer from the coordinator URLs' hostname pattern.
    # e.g. coordinators are "memgraph-coordinator-1" and data is
    # "memgraph-data-0.default.svc.cluster.local" → just use "memgraph-data-0".
    return short


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

        # HA support:
        # - Single neo4j:// URL: use native bolt+routing (driver handles failover).
        # - Multiple bolt:// URLs: coordinator discovery via SHOW INSTANCES.
        # - Single bolt:// URL: direct connection (standalone mode).
        urls = [u.strip() for u in url.split(",") if u.strip()]
        primary_scheme = urlparse(urls[0]).scheme

        if primary_scheme.startswith("neo4j"):
            # Native routing mode — pass through directly.
            target_url = urls[0]
            self._ha_coordinators = None
        elif len(urls) > 1:
            # Coordinator discovery mode.
            target_url = _discover_main(urls)
            self._ha_coordinators = urls
        else:
            target_url = urls[0]
            self._ha_coordinators = None

        self.driver = GraphDatabase.driver(target_url, auth=(username, password), **config)
        self._auth = (username, password)
        self._driver_config = config

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
        from neo4j.exceptions import ServiceUnavailable

        if params is None:
            params = {}
        try:
            return self._execute(query, params)
        except (ServiceUnavailable, OSError):
            if not self._ha_coordinators:
                raise
            # HA failover: MAIN likely changed, re-discover and retry.
            self._reconnect_to_main()
            return self._execute(query, params)

    def _execute(self, query: str, params: dict) -> list[dict[str, Any]]:
        """Run a query, falling back to implicit transactions when needed."""
        from neo4j.exceptions import Neo4jError

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

    def _reconnect_to_main(self) -> None:
        """Re-discover MAIN from coordinators and reconnect."""
        self.driver.close()
        target_url = _discover_main(self._ha_coordinators)
        self.driver = GraphDatabase.driver(target_url, auth=self._auth, **self._driver_config)

    def close(self) -> None:
        """
        Close the database connection.
        """
        self.driver.close()
