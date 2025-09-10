"""
DeepEval model resolver implementation.
"""

from typing import Optional

from deepeval.models import OllamaModel, LocalModel

from .base import BaseModelResolver


class DeepEvalModelResolver(BaseModelResolver):
    """
    Resolves model names and configurations for DeepEval.
    """

    def __init__(self, model_name: str, base_url: str = None):
        super().__init__(model_name, base_url)

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
        if model_name.startswith("vllm/"):  # E.g. vllm/llama-3.2-3b-instruct
            return model_name.split("vllm/", 1)[1]
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
            if self.base_url is None:
                return {"base_url": "http://localhost:11434", "temperature": 0}
            else:
                return {"base_url": self.base_url, "temperature": 0}
        if "qwen" in model_name:
            # TODO(gitbuda): That's another point, it actually depends how do
            # you deploy the model, vllm vs ollama is different. Probably add
            # the base_url as an argument to the constructor.
            return {"api_base": "http://muhlo:8000/v1"}
        if model_name.startswith("vllm/"):
            return {
                "base_url": "http://muhlo:8000/v1/",
                "api_key": "not-needed",
            }
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
        elif self.model_name.startswith("vllm/"):
            return LocalModel(
                model=self.get_model_name(),
                **self.get_model_config(),
            )
        else:
            return None
