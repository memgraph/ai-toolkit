from typing import Dict, List
from .tool import BaseTool


class BaseToolkit:
    """
    A Toolkit for managing tools.
    """

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def add_tool(self, tool: BaseTool) -> None:
        """
        Add a new tool in the registry.

        Raises:
            ValueError: If a tool with the same name is already toolkit.
        """
        if tool.name in self._tools:
            raise ValueError(
                f"Tool with name '{tool.name}' is already present in toolkit."
            )
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> BaseTool:
        """
        Retrieve a tool by its name.

        Raises:
            ValueError: If no tool with the given name is found.
        """
        if name not in self._tools:
            raise ValueError(f"Tool with name '{name}' not found.")
        return self._tools[name]

    def get_all_tools(self) -> List[BaseTool]:
        """
        List all added tools.
        """
        return list(self._tools.values())
