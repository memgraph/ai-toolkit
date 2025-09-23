"""Unit tests for the CoherenceEmbeddingsBasedMetric class."""

import pytest
from deepeval.test_case import LLMTestCase

from memgraph_toolbox.evals.coherence import (
    CoherenceEmbeddingsBasedMetric,
    evaluate_text_coherence,
)


class TestCoherenceEmbeddingsBasedMetric:
    """Test cases for CoherenceEmbeddingsBasedMetric."""

    def test_initialization(self):
        """Test metric initialization with default parameters."""
        metric = CoherenceEmbeddingsBasedMetric()

        assert metric.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert metric.min_sentences == 2
        assert metric.max_sentences == 100
        assert metric.threshold == 0.6
        assert metric.async_mode is True
        assert metric.score is None
        assert metric.success is None

    def test_initialization_custom_params(self):
        """Test metric initialization with custom parameters."""
        metric = CoherenceEmbeddingsBasedMetric(
            model_name="custom-model",
            min_sentences=3,
            max_sentences=50,
            threshold=0.7,
            async_mode=False,
        )

        assert metric.model_name == "custom-model"
        assert metric.min_sentences == 3
        assert metric.max_sentences == 50
        assert metric.threshold == 0.7
        assert metric.async_mode is False

    def test_async_mode_property(self):
        """Test async_mode property getter and setter."""
        metric = CoherenceEmbeddingsBasedMetric()

        assert metric.async_mode is True

        metric.async_mode = False
        assert metric.async_mode is False

    def test_split_into_sentences(self):
        """Test sentence splitting functionality."""
        metric = CoherenceEmbeddingsBasedMetric()

        text = "First sentence. Second sentence! Third sentence? Fourth sentence."
        sentences = metric._split_into_sentences(text)

        expected = [
            "First sentence.",
            "Second sentence!",
            "Third sentence?",
            "Fourth sentence.",
        ]
        assert sentences == expected

    def test_split_into_sentences_with_short_sentences(self):
        """Test that very short sentences are filtered out."""
        metric = CoherenceEmbeddingsBasedMetric()

        text = "Hi. This is a longer sentence. OK. Another longer sentence here."
        sentences = metric._split_into_sentences(text)

        # Should filter out "Hi." and "OK." as they're too short
        expected = ["This is a longer sentence.", "Another longer sentence here."]
        assert sentences == expected

    def test_split_into_sentences_empty_text(self):
        """Test sentence splitting with empty text."""
        metric = CoherenceEmbeddingsBasedMetric()

        sentences = metric._split_into_sentences("")
        assert sentences == []

    def test_split_into_sentences_whitespace_normalization(self):
        """Test that extra whitespace is normalized."""
        metric = CoherenceEmbeddingsBasedMetric()

        text = "First sentence.    Second sentence!   \n\nThird sentence?"
        sentences = metric._split_into_sentences(text)

        expected = ["First sentence.", "Second sentence!", "Third sentence?"]
        assert sentences == expected

    def test_measure_coherent_text(self):
        """Test measuring coherence of coherent text."""
        metric = CoherenceEmbeddingsBasedMetric()

        coherent_text = """
        Artificial intelligence is transforming the way we work and live. 
        Machine learning algorithms can now process vast amounts of data quickly. 
        This enables businesses to make better decisions based on data insights. 
        As a result, companies are becoming more efficient and competitive.
        """

        test_case = LLMTestCase(
            input="dummy", actual_output=coherent_text, expected_output="dummy"
        )
        score = metric.measure(test_case)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert metric.score == score
        assert isinstance(metric.success, bool)

    def test_measure_incoherent_text(self):
        """Test measuring coherence of incoherent text."""
        metric = CoherenceEmbeddingsBasedMetric()

        incoherent_text = """
        The weather is nice today. 
        Quantum physics involves complex mathematical equations. 
        I had pizza for lunch yesterday. 
        Machine learning requires large datasets for training.
        """

        test_case = LLMTestCase(
            input="dummy", actual_output=incoherent_text, expected_output="dummy"
        )
        score = metric.measure(test_case)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert metric.score == score
        assert isinstance(metric.success, bool)

    def test_measure_empty_text(self):
        """Test measuring coherence of empty text."""
        metric = CoherenceEmbeddingsBasedMetric()

        test_case = LLMTestCase(
            input="dummy", actual_output="", expected_output="dummy"
        )
        score = metric.measure(test_case)

        assert score == 0.0
        assert metric.score == 0.0
        assert metric.success is False

    def test_measure_none_text(self):
        """Test measuring coherence of None text."""
        metric = CoherenceEmbeddingsBasedMetric()

        test_case = LLMTestCase(
            input="dummy", actual_output=None, expected_output="dummy"
        )
        score = metric.measure(test_case)

        assert score == 0.0
        assert metric.score == 0.0
        assert metric.success is False

    def test_measure_short_text(self):
        """Test measuring coherence of text with insufficient sentences."""
        metric = CoherenceEmbeddingsBasedMetric()

        short_text = "Hello world."
        test_case = LLMTestCase(
            input="dummy", actual_output=short_text, expected_output="dummy"
        )
        score = metric.measure(test_case)

        assert score == 0.0
        assert metric.score == 0.0
        assert metric.success is False

    def test_measure_text_too_many_sentences(self):
        """Test that text with too many sentences is truncated."""
        metric = CoherenceEmbeddingsBasedMetric(max_sentences=3)

        # Create text with more than 3 sentences
        long_text = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."
        test_case = LLMTestCase(
            input="dummy", actual_output=long_text, expected_output="dummy"
        )
        score = metric.measure(test_case)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_is_successful(self):
        """Test is_successful method."""
        metric = CoherenceEmbeddingsBasedMetric(threshold=0.5)

        # Initially should be None
        assert metric.success is None

        # After measurement, should return boolean
        test_case = LLMTestCase(
            input="dummy",
            actual_output="Test sentence. Another test sentence.",
            expected_output="dummy",
        )
        metric.measure(test_case)

        assert isinstance(metric.is_successful(), bool)

    def test_metric_name(self):
        """Test metric name property."""
        metric = CoherenceEmbeddingsBasedMetric()
        assert metric.__name__ == "CoherenceEmbeddings"

    def test_a_measure_async(self):
        """Test async measure method."""
        metric = CoherenceEmbeddingsBasedMetric()

        test_case = LLMTestCase(
            input="dummy",
            actual_output="Test sentence. Another test sentence.",
            expected_output="dummy",
        )

        # Should be able to call async method
        import asyncio

        score = asyncio.run(metric.a_measure(test_case))

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


class TestEvaluateTextCoherence:
    """Test cases for the evaluate_text_coherence convenience function."""

    def test_evaluate_coherent_text(self):
        """Test evaluating coherent text."""
        coherent_text = """
        Artificial intelligence is transforming the way we work and live. 
        Machine learning algorithms can now process vast amounts of data quickly. 
        This enables businesses to make better decisions based on data insights. 
        As a result, companies are becoming more efficient and competitive.
        """

        result = evaluate_text_coherence(coherent_text)

        assert "coherence_score" in result
        assert "is_successful" in result
        assert "model_name" in result
        assert "min_sentences" in result
        assert "max_sentences" in result

        assert isinstance(result["coherence_score"], float)
        assert 0.0 <= result["coherence_score"] <= 1.0
        assert isinstance(result["is_successful"], bool)
        assert result["model_name"] == "sentence-transformers/all-MiniLM-L6-v2"
        assert result["min_sentences"] == 2
        assert result["max_sentences"] == 100

    def test_evaluate_incoherent_text(self):
        """Test evaluating incoherent text."""
        incoherent_text = """
        The weather is nice today. 
        Quantum physics involves complex mathematical equations. 
        I had pizza for lunch yesterday. 
        Machine learning requires large datasets for training.
        """

        result = evaluate_text_coherence(incoherent_text)

        assert "coherence_score" in result
        assert "is_successful" in result
        assert isinstance(result["coherence_score"], float)
        assert 0.0 <= result["coherence_score"] <= 1.0
        assert isinstance(result["is_successful"], bool)

    def test_evaluate_empty_text(self):
        """Test evaluating empty text."""
        result = evaluate_text_coherence("")

        assert result["coherence_score"] == 0.0
        assert result["is_successful"] is False

    def test_evaluate_short_text(self):
        """Test evaluating text with insufficient sentences."""
        short_text = "Hello world."
        result = evaluate_text_coherence(short_text)

        assert result["coherence_score"] == 0.0
        assert result["is_successful"] is False

    def test_evaluate_with_custom_params(self):
        """Test evaluating with custom parameters."""
        text = "First sentence. Second sentence."
        result = evaluate_text_coherence(
            text, model_name="custom-model", min_sentences=2, max_sentences=50
        )

        assert result["model_name"] == "custom-model"
        assert result["min_sentences"] == 2
        assert result["max_sentences"] == 50
