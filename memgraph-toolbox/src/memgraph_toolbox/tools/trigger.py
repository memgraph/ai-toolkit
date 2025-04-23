from typing import Any, Dict, List

from ..api.memgraph import Memgraph
from ..api.tool import BaseTool


class ShowTriggersTool(BaseTool):
    """
    Tool for showing trigger information from Memgraph.
    """

    def __init__(self, db: Memgraph):
        super().__init__(
            name="show_triggers",
            description="Shows trigger information from a Memgraph database",
            input_schema={"type": "object", "properties": {}, "required": []},
        )
        self.db = db

    def call(self, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute the SHOW TRIGGERS query and return the results."""
        trigger_info = self.db.query("SHOW TRIGGERS")
        return trigger_info
