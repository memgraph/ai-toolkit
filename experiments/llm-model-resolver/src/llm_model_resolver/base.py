"""
Base model resolver abstract class.
"""

from abc import ABC, abstractmethod


# TODO(gitbud): Make the openrouer resolver (add it under the existing resolvers).


class BaseModelResolver(ABC):
    """
    Abstract base class for resolving model names and their configuration.
    """

    def __init__(self, model_name: str, base_url: str = None):
        """
        Initialize the resolver with the project-specific model name.

        Args:
            model_name: The model name to resolve
        """
        self.model_name = model_name
        self.base_url = base_url

    @abstractmethod
    def get_model_name(self) -> str:
        """
        Return the resolved model name as a string.

        Returns:
            The resolved model name
        """
        pass

    @abstractmethod
    def get_model_config(self) -> dict:
        """
        Return the model configuration as a dictionary.

        Returns:
            Dictionary containing model configuration
        """
        pass

    @abstractmethod
    def get_model(self):
        """
        Return the model instance.

        Returns:
            The model instance or None if not applicable
        """
        pass
