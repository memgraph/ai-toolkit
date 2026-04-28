from .tool import BaseTool


class BaseToolbox:
    """
    A Toolbox for managing tools.
    """

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def add_tool(self, tool: BaseTool) -> None:
        """
        Add a new tool in the registry.

        Raises:
            ValueError: If a tool with the same name is already in toolbox.
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool with name '{tool.name}' is already present in toolbox.")
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

    def get_all_tools(self) -> list[BaseTool]:
        """
        List all added tools.
        """
        return list(self._tools.values())
