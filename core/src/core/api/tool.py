from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseTool(ABC):
    """
    Base class for all tools.
    """

    name: str
    description: str
    input_schema: Dict[str, Any]

    def __init__(self, name: str, description: str, input_schema: Dict[str, Any]):
        self.name = name
        self.description = description
        self.input_schema = input_schema

    @abstractmethod
    def call(self, arguments: Dict[str, Any]) -> List[Any]:
        """
        Execute the tool with the provided arguments.

        Parameters:
            arguments (dict): A dictionary of arguments as defined by the input schema.

        Returns:
            List containing one or more output values.
        """
        pass

    def __repr__(self):
        return f"Tool(name={self.name}, description={self.description})"

    def get_name(self) -> str:
        """
        Get the name of the tool.

        Returns:
            str: The name of the tool.
        """
        return self.name

    def get_description(self) -> str:
        """
        Get the description of the tool.

        Returns:
            str: The description of the tool.
        """
        return self.description
