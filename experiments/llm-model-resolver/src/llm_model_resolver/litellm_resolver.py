"""
LiteLLM model resolver implementation.
"""

from typing import Optional

from .base import BaseModelResolver


class LiteLLMModelResolver(BaseModelResolver):
    """
    Resolves model names and configurations for LiteLLM, supporting Ollama and OpenAI models.
    """

    def __init__(self, model_name: str, base_url: str = None):
        super().__init__(model_name, base_url)

    def get_model_name(self) -> str:
        """
        Returns the resolved model name for LiteLLM.
        For Ollama models, returns the model name as is.
        For OpenAI models, returns the OpenAI model name.

        Returns:
            The resolved model name for LiteLLM
        """
        model_name = self.model_name.lower()
        if model_name.startswith("ollama/"):
            return model_name
        elif model_name in ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o"]:
            return model_name
        elif model_name.startswith("gpt-"):
            return model_name
        else:
            return model_name

    def get_model_config(self) -> dict:
        """
        Returns the model configuration for LiteLLM.
        For Ollama, returns a dict with 'api_base' key.
        For OpenAI, returns an empty dict (LiteLLM expects OpenAI API key in env).

        Returns:
            Dictionary containing model configuration
        """
        model_name = self.model_name.lower()
        if model_name.startswith("ollama/"):
            if self.base_url is None:
                return {"api_base": "http://localhost:11434"}
            else:
                return {"api_base": self.base_url}
        elif model_name in [
            "gpt-3.5-turbo",
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4o",
        ] or model_name.startswith("gpt-"):
            return {}
        else:
            return {}

    def get_model(self) -> Optional[None]:
        """
        Returns a tuple of (model_name, model_config) for LiteLLM.

        Returns:
            None (LiteLLM doesn't require a model instance)
        """
        return None
