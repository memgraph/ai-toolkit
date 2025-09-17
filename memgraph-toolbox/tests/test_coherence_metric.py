"""Tests for the coherence embeddings metric."""

import pytest
from memgraph_toolbox.evals.coherence_embeddings_based import (
    CoherenceEmbeddingsMetric,
    evaluate_text_coherence,
)
from deepeval.test_case import LLMTestCase


class TestCoherenceEmbeddingsMetric:
    """Test cases for the CoherenceEmbeddingsMetric class."""

    def test_metric_initialization(self):
        """Test that the metric initializes correctly."""
        metric = CoherenceEmbeddingsMetric()
        assert metric.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert metric.min_sentences == 2
        assert metric.max_sentences == 100

    def test_metric_initialization_custom_params(self):
        """Test metric initialization with custom parameters."""
        metric = CoherenceEmbeddingsMetric(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            min_sentences=3,
            max_sentences=50,
        )
        assert metric.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert metric.min_sentences == 3
        assert metric.max_sentences == 50

    def test_split_into_sentences(self):
        """Test sentence splitting functionality."""
        metric = CoherenceEmbeddingsMetric()

        text = "This is the first sentence. This is the second sentence! And this is the third sentence?"
        sentences = metric._split_into_sentences(text)

        assert len(sentences) == 3
        assert "This is the first sentence" in sentences[0]
        assert "This is the second sentence" in sentences[1]
        assert "And this is the third sentence" in sentences[2]

    def test_split_into_sentences_short_text(self):
        """Test sentence splitting with very short text."""
        metric = CoherenceEmbeddingsMetric()

        text = "Short."
        sentences = metric._split_into_sentences(text)

        # Should filter out very short sentences
        assert len(sentences) == 0

    def test_measure_coherent_text(self):
        """Test measuring coherence of a coherent text."""
        metric = CoherenceEmbeddingsMetric()

        coherent_text = """
        Artificial intelligence is transforming industries worldwide. 
        Machine learning algorithms enable computers to learn from data. 
        This technology is being applied in healthcare, finance, and transportation. 
        As a result, we are seeing significant improvements in efficiency and accuracy.
        """

        test_case = LLMTestCase(
            input="What is AI?",
            actual_output=coherent_text,
            expected_output="AI explanation",
        )

        score = metric.measure(test_case)
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should be reasonably coherent

    def test_measure_incoherent_text(self):
        """Test measuring coherence of an incoherent text."""
        metric = CoherenceEmbeddingsMetric()

        incoherent_text = """
        The weather is sunny today. 
        Quantum physics involves complex mathematics. 
        I had pizza for lunch. 
        Machine learning requires data.
        """

        test_case = LLMTestCase(
            input="Random topics",
            actual_output=incoherent_text,
            expected_output="Random response",
        )

        score = metric.measure(test_case)
        assert 0.0 <= score <= 1.0
        # Should be less coherent than the coherent text
        assert score < 0.8

    def test_measure_insufficient_sentences(self):
        """Test measuring text with insufficient sentences."""
        metric = CoherenceEmbeddingsMetric(min_sentences=5)

        short_text = "This is a short text. It has only two sentences."

        test_case = LLMTestCase(
            input="Short text",
            actual_output=short_text,
            expected_output="Short response",
        )

        score = metric.measure(test_case)
        assert score == 0.0  # Should return 0 for insufficient sentences

    def test_measure_empty_text(self):
        """Test measuring empty text."""
        metric = CoherenceEmbeddingsMetric()

        test_case = LLMTestCase(
            input="Empty", actual_output="", expected_output="Empty"
        )

        score = metric.measure(test_case)
        assert score == 0.0

    def test_measure_non_string_input(self):
        """Test measuring non-string input."""
        metric = CoherenceEmbeddingsMetric()

        test_case = LLMTestCase(
            input="Non-string",
            actual_output=123,  # Non-string
            expected_output="Number",
        )

        score = metric.measure(test_case)
        assert score == 0.0

    def test_is_successful(self):
        """Test the is_successful method."""
        metric = CoherenceEmbeddingsMetric()

        # Test with default threshold (0.6)
        assert metric.is_successful(0.7) is True
        assert metric.is_successful(0.5) is False

        # Test with custom threshold
        assert metric.is_successful(0.8, threshold=0.9) is False
        assert metric.is_successful(0.95, threshold=0.9) is True

    def test_metric_name(self):
        """Test the metric name property."""
        metric = CoherenceEmbeddingsMetric()
        assert metric.__name__ == "CoherenceEmbeddings"


class TestEvaluateTextCoherence:
    """Test cases for the evaluate_text_coherence convenience function."""

    def test_evaluate_coherent_text(self):
        """Test evaluating coherent text."""
        coherent_text = """
        The sun rises in the east every morning. 
        This natural phenomenon occurs due to Earth's rotation. 
        As a result, we experience day and night cycles. 
        This rotation also affects weather patterns globally.
        """

        result = evaluate_text_coherence(coherent_text)

        assert "coherence_score" in result
        assert "is_successful" in result
        assert "model_name" in result
        assert "min_sentences" in result
        assert "max_sentences" in result

        assert 0.0 <= result["coherence_score"] <= 1.0
        assert isinstance(result["is_successful"], bool)
        assert result["model_name"] == "sentence-transformers/all-MiniLM-L6-v2"

    def test_evaluate_with_custom_params(self):
        """Test evaluating with custom parameters."""
        text = "First sentence. Second sentence. Third sentence."

        result = evaluate_text_coherence(
            text,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            min_sentences=2,
            max_sentences=10,
        )

        assert result["min_sentences"] == 2
        assert result["max_sentences"] == 10


if __name__ == "__main__":
    # Run a simple test
    print("Running coherence metric test...")

    # Test with coherent text
    coherent_text = """
    Machine learning is a subset of artificial intelligence. 
    It enables computers to learn patterns from data without explicit programming. 
    This technology is revolutionizing many industries today. 
    As a result, we are seeing more intelligent and automated systems.
    """

    result = evaluate_text_coherence(coherent_text)
    print(f"Coherent text score: {result['coherence_score']:.3f}")
    print(f"Is successful: {result['is_successful']}")

    # Test with incoherent text
    incoherent_text = """
    The sky is blue today. 
    Mathematics involves numbers and equations. 
    I like to eat ice cream. 
    Computers process information using binary code.
    """

    result = evaluate_text_coherence(incoherent_text)
    print(f"Incoherent text score: {result['coherence_score']:.3f}")
    print(f"Is successful: {result['is_successful']}")

    print("Test completed!")
