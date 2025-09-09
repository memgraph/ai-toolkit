"""
DeepEval model resolver implementation.
"""

from typing import Optional

from deepeval.models import OllamaModel

from .base import BaseModelResolver


class DeepEvalModelResolver(BaseModelResolver):
    """
    Resolves model names and configurations for DeepEval.
    """

    def __init__(self, model_name: str):
        super().__init__(model_name)

    def get_model_name(self) -> str:
        """
        Returns the resolved model name for DeepEval.

        Returns:
            The resolved model name for DeepEval
        """
        model_name = self.model_name.lower()
        if model_name.startswith("ollama/"):
            # e.g., "ollama:llama3.2:latest" -> "llama3.2:latest"
            return model_name.split("ollama/", 1)[1]
        elif model_name in ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o"]:
            return model_name
        elif model_name.startswith("gpt-"):
            return model_name
        else:
            return model_name

    def get_model_config(self) -> dict:
        """
        Returns the model configuration for DeepEval.

        Returns:
            Dictionary containing model configuration
        """
        model_name = self.model_name.lower()
        if model_name.startswith("ollama/"):
            return {"base_url": "http://localhost:11434"}
        elif model_name in [
            "gpt-3.5-turbo",
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4o",
        ] or model_name.startswith("gpt-"):
            return {}
        else:
            return {}

    def get_model(self) -> Optional[OllamaModel]:
        """
        Returns the DeepEval model instance.

        Returns:
            OllamaModel instance for Ollama models, None for others
        """
        if self.model_name.startswith("ollama/"):
            return OllamaModel(
                model=self.get_model_name(),
                **self.get_model_config(),
            )
        else:
            return None
