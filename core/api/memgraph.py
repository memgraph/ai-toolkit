from typing import Any, Dict, List
from neo4j import GraphDatabase


class MemgraphClient:
    """
    Base Memgraph client for interacting with Memgraph database.
    """
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def query(self, query: str) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query and return results as a list of dictionaries.
        
        Args:
            query: The Cypher query to execute
            
        Returns:
            List of dictionaries containing query results
        """
        with self.driver.session() as session:
            result = session.run(query)
            return [dict(record) for record in result]

    def close(self) -> None:
        """
        Close the database connection.
        """
        self.driver.close() 