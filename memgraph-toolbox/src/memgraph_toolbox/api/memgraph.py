import logging
import os
from collections.abc import Mapping
from typing import Any

from neo4j import GraphDatabase

from ..utils.serialization import serialize_record_data

logger = logging.getLogger(__name__)

MEMGRAPH_ENV_DEFAULTS = {
    "MEMGRAPH_URL": "bolt://localhost:7687",
    "MEMGRAPH_USER": "",
    "MEMGRAPH_PASSWORD": "",
    "MEMGRAPH_DATABASE": "memgraph",
    "MEMGRAPH_HA_CLUSTER": "false",
}
MEMGRAPH_ENV_KEYS = tuple(MEMGRAPH_ENV_DEFAULTS)

# URL schemes that enable Bolt+routing (auto-routing to the current MAIN in an
# HA cluster). See:
# https://memgraph.com/docs/clustering/high-availability/querying-the-cluster-in-high-availability
_ROUTING_SCHEMES = ("neo4j://", "neo4j+s://", "neo4j+ssc://")
_BOLT_TO_ROUTING_SCHEME = {
    "bolt://": "neo4j://",
    "bolt+s://": "neo4j+s://",
    "bolt+ssc://": "neo4j+ssc://",
}


def _is_committed_on_main_error(message: str) -> bool:
    """True for an HA commit that reached the MAIN but failed to replicate to a SYNC replica.

    In a Memgraph high-availability cluster, if a SYNC replica is momentarily unavailable,
    the commit raises an error even though the write IS committed on the MAIN (and other
    alive replicas), and the lagging replica is recovered automatically. Aborting the caller
    would be wrong: the data is durably committed. An idempotent client should tolerate this.
    See https://memgraph.com/docs/clustering/high-availability
    """
    return "Replication Exception" in message and "committed on the main" in message


def _as_bool(value: str | bool | None) -> bool:
    """Interpret a truthy string/bool flag (e.g. from an env var)."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return value.strip().lower() in ("1", "true", "yes", "on")


def to_routing_url(url: str) -> str:
    """Return ``url`` using a Bolt+routing scheme so the driver routes to the MAIN.

    A ``bolt://`` URL pointed at the coordinators is upgraded to ``neo4j://``;
    URLs that already use a routing scheme are returned unchanged.
    """
    if url.startswith(_ROUTING_SCHEMES):
        return url
    for bolt_scheme, routing_scheme in _BOLT_TO_ROUTING_SCHEME.items():
        if url.startswith(bolt_scheme):
            return routing_scheme + url[len(bolt_scheme) :]
    return url


def memgraph_env(
    *,
    url: str | None = None,
    username: str | None = None,
    password: str | None = None,
    database: str | None = None,
    ha_cluster: bool | None = None,
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
        "MEMGRAPH_HA_CLUSTER": (
            ("true" if ha_cluster else "false")
            if ha_cluster is not None
            else env.get("MEMGRAPH_HA_CLUSTER", MEMGRAPH_ENV_DEFAULTS["MEMGRAPH_HA_CLUSTER"])
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
        ha_cluster: bool | None = None,
    ):
        """
        Initialize Memgraph client with connection parameters.

        Connection details can be provided directly or via environment variables:
        - MEMGRAPH_URL (default: "bolt://localhost:7687")
        - MEMGRAPH_USER (default: "")
        - MEMGRAPH_PASSWORD (default: "")
        - MEMGRAPH_DATABASE (default: "memgraph")
        - MEMGRAPH_HA_CLUSTER (default: "false")

        Args:
            url: The Memgraph connection URL
            username: Username for authentication
            password: Password for authentication
            database: The database name to connect to (default: "memgraph")
            driver_config: Additional Neo4j driver configuration
            user_agent: Client name sent to the server (e.g. "mcp-memgraph", "langchain-memgraph", "sql2graph", etc.)
            ha_cluster: Connect to a Memgraph high-availability cluster. When enabled,
                the URL should point at the coordinators (e.g. "neo4j://mg-coordinators:7687")
                and Bolt+routing is used so queries are automatically routed to the current
                MAIN instance, surviving failovers. A "bolt://" URL is upgraded to "neo4j://".
                See https://memgraph.com/docs/clustering/high-availability/querying-the-cluster-in-high-availability
        """

        # Load from the shared Memgraph environment contract with fallbacks.
        env = memgraph_env(
            url=url, username=username, password=password, database=database, ha_cluster=ha_cluster
        )
        url = env["MEMGRAPH_URL"]
        username = env["MEMGRAPH_USER"]
        password = env["MEMGRAPH_PASSWORD"]

        # In HA mode, use Bolt+routing so the driver always reaches the current MAIN
        # through the coordinators and refreshes its routing table on failover.
        self.ha_cluster = _as_bool(env["MEMGRAPH_HA_CLUSTER"])
        if self.ha_cluster:
            url = to_routing_url(url)

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

        In an HA cluster, a commit that reaches the MAIN but fails to replicate to a
        momentarily-unavailable SYNC replica is tolerated (logged as a warning) because
        the write is already committed on the MAIN and the replica recovers automatically.

        Args:
            query: The Cypher query to execute

        Returns:
            List of dictionaries containing query results
        """
        from neo4j.exceptions import Neo4jError

        if params is None:
            params = {}
        try:
            return self._execute(query, params)
        except Neo4jError as e:
            if _is_committed_on_main_error(getattr(e, "message", "") or str(e)):
                logger.warning(
                    "Memgraph HA: commit reached the MAIN but a SYNC replica failed to "
                    "replicate; continuing since the write is committed on the MAIN and "
                    "the replica recovers automatically (%s)",
                    getattr(e, "message", "") or e,
                )
                return []
            raise

    def _execute(self, query: str, params: dict) -> list[dict[str, Any]]:
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

    def close(self) -> None:
        """
        Close the database connection.
        """
        self.driver.close()
