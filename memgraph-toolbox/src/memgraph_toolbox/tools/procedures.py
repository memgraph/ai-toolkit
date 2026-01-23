from typing import Any, Dict, List

from ..api.memgraph import Memgraph
from ..api.tool import BaseTool


class ShowProceduresTool(BaseTool):
    """
    Tool for listing all available Memgraph procedures (query modules).

    This tool queries the Memgraph database to retrieve information about
    all available procedures from MAGE and custom query modules.
    """

    def __init__(self, db: Memgraph):
        super().__init__(
            name="show_procedures",
            description=(
                "Lists all available Memgraph procedures (query modules) including "
                "MAGE algorithms and custom query modules. Returns procedure name, "
                "signature, and whether it's a read-only operation. Use this to "
                "discover available graph algorithms and utility functions."
            ),
            input_schema={"type": "object", "properties": {}, "required": []},
        )
        self.db = db

    def call(self, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute the mg.procedures() query and return the available procedures."""
        procedures = self.db.query(
            "CALL mg.procedures() YIELD name, signature, is_write "
            "RETURN name, signature, is_write "
            "ORDER BY name"
        )
        return procedures
