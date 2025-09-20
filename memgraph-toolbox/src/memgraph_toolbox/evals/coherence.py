"""Coherence evaluation metric based on embedding similarity between consecutive sentences."""

import re
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase


class CoherenceEmbeddingsBasedMetric(BaseMetric):
    """Coherence metric based on average cosine similarity between consecutive sentences.

    This metric evaluates text coherence by:
    1. Splitting text into sentences
    2. Computing embeddings for each sentence
    3. Calculating cosine similarity between consecutive sentences
    4. Returning the average similarity as the coherence score

    Higher scores indicate better coherence (more logical flow between sentences).
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        min_sentences: int = 2,
        max_sentences: int = 100,
        threshold: float = 0.6,
        async_mode: bool = True,
    ):
        """Initialize the coherence metric.

        Args:
            model_name: Name of the sentence transformer model to use
            min_sentences: Minimum number of sentences required for evaluation
            max_sentences: Maximum number of sentences to process (for performance)
            threshold: Threshold for determining success (default: 0.6)
            async_mode: Whether to run in async mode
        """
        super().__init__()
        self.model_name = model_name
        self.min_sentences = min_sentences
        self.max_sentences = max_sentences
        self.threshold = threshold
        self._async_mode = async_mode
        self._model = None
        self.score = None
        self.success = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the sentence transformer model."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def async_mode(self) -> bool:
        """Return whether the metric runs in async mode."""
        return self._async_mode

    @async_mode.setter
    def async_mode(self, value: bool) -> None:
        """Set async mode."""
        self._async_mode = value

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex patterns.

        Args:
            text: Input text to split

        Returns:
            List of sentences
        """
        # Remove extra whitespace and normalize
        text = re.sub(r"\s+", " ", text.strip())

        # Split on sentence boundaries (., !, ?) followed by whitespace and capital letter
        # or end of string
        sentence_pattern = r"(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\s*$"
        sentences = re.split(sentence_pattern, text)

        # Clean up sentences
        sentences = [s.strip() for s in sentences if s.strip()]

        # Filter out very short sentences (likely fragments)
        sentences = [s for s in sentences if len(s) > 10]

        return sentences

    def _compute_embeddings(self, sentences: List[str]) -> np.ndarray:
        """Compute embeddings for a list of sentences.

        Args:
            sentences: List of sentences

        Returns:
            Array of embeddings with shape (n_sentences, embedding_dim)
        """
        # TODO(gitbuda): This should use the utils.embeddings because GPU could be used.
        return self.model.encode(sentences)

    def _compute_cosine_similarity(self, embeddings: np.ndarray) -> List[float]:
        """Compute cosine similarity between consecutive embeddings.

        Args:
            embeddings: Array of embeddings with shape (n_sentences, embedding_dim)

        Returns:
            List of cosine similarities between consecutive sentence pairs
        """
        if len(embeddings) < 2:
            return []

        similarities = []
        for i in range(len(embeddings) - 1):
            # Compute cosine similarity between consecutive sentences
            emb1 = embeddings[i]
            emb2 = embeddings[i + 1]

            # Normalize embeddings
            emb1_norm = emb1 / np.linalg.norm(emb1)
            emb2_norm = emb2 / np.linalg.norm(emb2)

            # Compute cosine similarity
            similarity = np.dot(emb1_norm, emb2_norm)
            similarities.append(float(similarity))

        return similarities

    def measure(self, test_case: LLMTestCase) -> float:
        """Measure coherence of the actual output.

        Args:
            test_case: Test case containing the text to evaluate

        Returns:
            Coherence score between 0 and 1 (higher is better)
        """
        text = test_case.actual_output

        if not text or not isinstance(text, str):
            self.score = 0.0
            self.success = False
            return 0.0

        # Split into sentences
        sentences = self._split_into_sentences(text)

        # Check if we have enough sentences
        if len(sentences) < self.min_sentences:
            self.score = 0.0
            self.success = False
            return 0.0

        # Limit number of sentences for performance
        if len(sentences) > self.max_sentences:
            sentences = sentences[: self.max_sentences]

        try:
            # Compute embeddings
            embeddings = self._compute_embeddings(sentences)

            # Compute cosine similarities between consecutive sentences
            similarities = self._compute_cosine_similarity(embeddings)

            if not similarities:
                self.score = 0.0
                self.success = False
                return 0.0

            # Return average similarity as coherence score
            coherence_score = np.mean(similarities)

            # Normalize to 0-1 range (cosine similarity is already in -1 to 1 range)
            # We'll map it to 0-1 where 0.5 similarity becomes 0.75 coherence
            normalized_score = (coherence_score + 1) / 2

            # Store the score and success for DeepEval
            self.score = float(normalized_score)
            self.success = self.score >= self.threshold

            return float(normalized_score)

        except Exception as e:
            # If there's an error, return 0
            print(f"Error computing coherence: {e}")
            self.score = 0.0
            self.success = False
            return 0.0

    async def a_measure(self, test_case: "LLMTestCase") -> float:
        """Async version of measure for coherence evaluation.

        Args:
            test_case: Test case containing the text to evaluate

        Returns:
            Coherence score between 0 and 1 (higher is better)
        """
        # For now, just call the synchronous version
        # In the future, this could be made truly async for better performance
        return self.measure(test_case)

    def is_successful(self) -> bool:
        """Determine if the coherence score meets the threshold.

        Returns:
            True if score meets threshold, False otherwise
        """
        return self.success

    @property
    def __name__(self) -> str:
        """Return the name of the metric."""
        return "CoherenceEmbeddings"


def evaluate_text_coherence(
    text: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    min_sentences: int = 2,
    max_sentences: int = 100,
) -> dict:
    """Convenience function to evaluate text coherence.

    Args:
        text: Text to evaluate
        model_name: Sentence transformer model to use
        min_sentences: Minimum sentences required
        max_sentences: Maximum sentences to process

    Returns:
        Dictionary with coherence score and metadata
    """
    metric = CoherenceEmbeddingsBasedMetric(
        model_name=model_name, min_sentences=min_sentences, max_sentences=max_sentences
    )

    # Create a dummy test case
    test_case = LLMTestCase(input="dummy", actual_output=text, expected_output="dummy")

    score = metric.measure(test_case)
    is_successful = metric.is_successful()

    return {
        "coherence_score": score,
        "is_successful": is_successful,
        "model_name": model_name,
        "min_sentences": min_sentences,
        "max_sentences": max_sentences,
    }

if __name__ == "__main__":
    example_text = (
        "Artificial intelligence is transforming industries. "
        "Machine learning enables computers to learn from data. "
        "This leads to smarter applications and better user experiences."
    )
    result = evaluate_text_coherence(example_text)
    print("Coherence Evaluation Result:")
    for k, v in result.items():
        print(f"{k}: {v}")
