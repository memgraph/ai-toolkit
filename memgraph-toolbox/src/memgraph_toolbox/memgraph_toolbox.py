from .api.memgraph import Memgraph
from .api.toolbox import BaseToolbox

# Import all tool classes
from .tools.cypher import CypherTool
from .tools.schema import EnumSchemaTool, NodeSchemaTool, RelationshipSchemaTool, SearchSchemaTool


class MemgraphToolbox(BaseToolbox):
    """
    A toolbox that contains all available Memgraph tools.
    This class extends the BaseToolbox to provide a convenient way to
    access all Memgraph-related tools.
    """

    def __init__(self, db: Memgraph):
        """
        Initialize the Memgraph toolbox with all available tools.

        Args:
            db: Memgraph database connection instance. If not provided,
                tools will need to be initialized with a connection separately.
        """
        super().__init__()

        if db is not None:
            self.add_tool(CypherTool(db))
            self.add_tool(SearchSchemaTool(db))
            self.add_tool(NodeSchemaTool(db))
            self.add_tool(RelationshipSchemaTool(db))
            self.add_tool(EnumSchemaTool(db))
        else:
            raise ValueError("Memgraph database connection is required to initialize tools.")
