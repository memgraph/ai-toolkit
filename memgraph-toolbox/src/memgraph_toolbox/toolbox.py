from typing import Optional


from .api.memgraph import Memgraph
from .api.toolkit import BaseToolkit

# Import all tool classes
from .tools.betweenness_centrality import BetweennessCentralityTool
from .tools.config import ShowConfigTool
from .tools.constraint import ShowConstraintInfoTool
from .tools.cypher import CypherTool
from .tools.index import ShowIndexInfoTool
from .tools.page_rank import PageRankTool
from .tools.schema import ShowSchemaInfoTool
from .tools.storage import ShowStorageInfoTool
from .tools.trigger import ShowTriggersTool


class MemgraphToolkit(BaseToolkit):
    """
    A toolbox that contains all available Memgraph tools.
    This class extends the BaseToolkit to provide a convenient way to
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
            self.add_tool(BetweennessCentralityTool(db))
            self.add_tool(ShowConfigTool(db))
            self.add_tool(ShowConstraintInfoTool(db))
            self.add_tool(CypherTool(db))
            self.add_tool(ShowIndexInfoTool(db))
            self.add_tool(PageRankTool(db))
            self.add_tool(ShowSchemaInfoTool(db))
            self.add_tool(ShowStorageInfoTool(db))
            self.add_tool(ShowTriggersTool(db))
        else:
            raise ValueError(
                "Memgraph database connection is required to initialize tools."
            )
