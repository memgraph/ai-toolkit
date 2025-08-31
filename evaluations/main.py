"""
Knowledge Retrieval Evaluation Tools

This module provides stub functionality for evaluating knowledge retrieval
responses in different scenarios. The aim is to design tools to be LLM agnostic
and easily extensible
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import uuid
from datetime import datetime


class EvaluationScore(Enum):
    """Enum for evaluation scores"""

    VERY_POOR = 1
    POOR = 2
    FAIR = 3
    GOOD = 4
    EXCELLENT = 5


@dataclass
class TestCase:
    """Represents a test case for evaluation"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    question: str = ""
    answer: str = ""
    context: List[str] = field(default_factory=list)
    ground_truth: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class EvaluationResult:
    """Result of an evaluation"""

    test_case_id: str
    metric_name: str
    score: EvaluationScore
    score_numeric: float
    reasoning: str
    passed: bool
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TestRun:
    """Represents a complete test run with multiple test cases"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    test_cases: List[TestCase] = field(default_factory=list)
    results: List[EvaluationResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def add_test_case(self, test_case: TestCase):
        """Add a test case to the test run"""
        self.test_cases.append(test_case)

    def add_result(self, result: EvaluationResult):
        """Add an evaluation result to the test run"""
        self.results.append(result)

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the test run"""
        if not self.results:
            return {"status": "No results", "total_tests": 0}

        total_tests = len(self.test_cases)

        # Group results by test case to properly calculate pass/fail
        test_case_results = {}
        for result in self.results:
            if result.test_case_id not in test_case_results:
                test_case_results[result.test_case_id] = []
            test_case_results[result.test_case_id].append(result)

        # Count passed/failed test cases (a test case passes if ALL its metrics pass)
        passed_test_cases = 0
        failed_test_cases = 0

        for test_case_id, results in test_case_results.items():
            # A test case passes only if ALL its metrics pass
            if all(result.passed for result in results):
                passed_test_cases += 1
            else:
                failed_test_cases += 1

        # Group results by metric for metric summary
        metric_results = {}
        for result in self.results:
            if result.metric_name not in metric_results:
                metric_results[result.metric_name] = []
            metric_results[result.metric_name].append(result)

        # Calculate average scores per metric
        metric_averages = {}
        for metric_name, results in metric_results.items():
            avg_score = sum(r.score_numeric for r in results) / len(results)
            metric_averages[metric_name] = {
                "average_score": avg_score,
                "total_tests": len(results),
                "passed": sum(1 for r in results if r.passed),
            }

        return {
            "status": "completed",
            "total_tests": total_tests,
            "passed_tests": passed_test_cases,
            "failed_tests": failed_test_cases,
            "pass_rate": passed_test_cases / total_tests if total_tests > 0 else 0,
            "metric_summary": metric_averages,
            "overall_average": sum(r.score_numeric for r in self.results)
            / len(self.results),
        }


class BaseMetric(ABC):
    """Base class for all evaluation metrics"""

    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold

    @abstractmethod
    def evaluate(self, test_case: TestCase) -> EvaluationResult:
        """Evaluate the test case and return a result"""
        pass

    def _score_to_enum(self, score: float) -> EvaluationScore:
        """Convert numeric score to enum"""
        if score >= 4.5:
            return EvaluationScore.EXCELLENT
        elif score >= 3.5:
            return EvaluationScore.GOOD
        elif score >= 2.5:
            return EvaluationScore.FAIR
        elif score >= 1.5:
            return EvaluationScore.POOR
        else:
            return EvaluationScore.VERY_POOR

    def _determine_pass(self, score: float) -> bool:
        """Determine if the test case passed based on threshold"""
        return score >= self.threshold


class AnswerRelevancyMetric(BaseMetric):
    """Evaluates how relevant the answer is to the question"""

    def __init__(self, threshold: float = 3.0):
        super().__init__(threshold)

    def evaluate(self, test_case: TestCase) -> EvaluationResult:
        """
        Stub implementation for answer relevancy evaluation.
        In a real implementation, this would use an LLM or other model to assess relevancy.
        """
        # Stub logic: simple keyword matching and length analysis
        question_lower = test_case.question.lower()
        answer_lower = test_case.answer.lower()

        # Count common words between question and answer
        question_words = set(question_lower.split())
        answer_words = set(answer_lower.split())
        common_words = question_words.intersection(answer_words)

        # Simple scoring based on word overlap and answer length
        word_overlap_ratio = len(common_words) / max(len(question_words), 1)
        length_score = min(
            len(test_case.answer) / 100, 1.0
        )  # Normalize by expected length

        # Combined score (stub implementation)
        score = (word_overlap_ratio * 0.7 + length_score * 0.3) * 5

        reasoning = f"Word overlap ratio: {word_overlap_ratio:.2f}, Length score: {length_score:.2f}"
        passed = self._determine_pass(score)

        return EvaluationResult(
            test_case_id=test_case.id,
            metric_name="answer_relevancy",
            score=self._score_to_enum(score),
            score_numeric=score,
            reasoning=reasoning,
            passed=passed,
            metadata={
                "word_overlap_ratio": word_overlap_ratio,
                "length_score": length_score,
                "common_words": list(common_words),
            },
        )


class FaithfulnessMetric(BaseMetric):
    """Evaluates how faithful the answer is to the provided context"""

    def __init__(self, threshold: float = 3.0):
        super().__init__(threshold)

    def evaluate(self, test_case: TestCase) -> EvaluationResult:
        """
        Stub implementation for faithfulness evaluation.
        In a real implementation, this would use an LLM to check if the answer
        is supported by the provided context.
        """
        # Stub logic: check if key phrases from context appear in answer
        context_text = " ".join(test_case.context).lower()
        answer_lower = test_case.answer.lower()

        # Extract key phrases from context (simple approach)
        context_words = set(context_text.split())
        answer_words = set(answer_lower.split())

        # Check for context coverage in answer
        context_coverage = len(answer_words.intersection(context_words)) / max(
            len(context_words), 1
        )

        # Check for potential hallucination (words in answer not in context)
        unique_answer_words = answer_words - context_words
        hallucination_penalty = min(
            len(unique_answer_words) / max(len(answer_words), 1), 0.5
        )

        # Calculate faithfulness score
        score = max(0, (context_coverage - hallucination_penalty)) * 5

        reasoning = f"Context coverage: {context_coverage:.2f}, Hallucination penalty: {hallucination_penalty:.2f}"
        passed = self._determine_pass(score)

        return EvaluationResult(
            test_case_id=test_case.id,
            metric_name="faithfulness",
            score=self._score_to_enum(score),
            score_numeric=score,
            reasoning=reasoning,
            passed=passed,
            metadata={
                "context_coverage": context_coverage,
                "hallucination_penalty": hallucination_penalty,
                "unique_answer_words": list(unique_answer_words),
            },
        )


class ContextualPrecisionMetric(BaseMetric):
    """Evaluates how relevant the retrieved context is to the question"""

    def __init__(self, threshold: float = 3.0):
        super().__init__(threshold)

    def evaluate(self, test_case: TestCase) -> EvaluationResult:
        """
        Stub implementation for contextual precision evaluation.
        In a real implementation, this would use an LLM to assess if the
        retrieved context is relevant to answering the question.
        """
        # Stub logic: analyze context relevance to question
        question_lower = test_case.question.lower()
        question_words = set(question_lower.split())

        total_relevance_score = 0
        context_scores = []

        for i, context_piece in enumerate(test_case.context):
            context_lower = context_piece.lower()
            context_words = set(context_lower.split())

            # Calculate relevance for this context piece
            word_overlap = len(question_words.intersection(context_words))
            relevance_score = word_overlap / max(len(question_words), 1)

            context_scores.append(
                {
                    "context_index": i,
                    "relevance_score": relevance_score,
                    "word_overlap": word_overlap,
                }
            )

            total_relevance_score += relevance_score

        # Average relevance across all context pieces
        avg_relevance = total_relevance_score / max(len(test_case.context), 1)
        final_score = avg_relevance * 5

        reasoning = f"Average context relevance: {avg_relevance:.2f} across {len(test_case.context)} context pieces"
        passed = self._determine_pass(final_score)

        return EvaluationResult(
            test_case_id=test_case.id,
            metric_name="contextual_precision",
            score=self._score_to_enum(final_score),
            score_numeric=final_score,
            reasoning=reasoning,
            passed=passed,
            metadata={
                "context_scores": context_scores,
                "total_relevance_score": total_relevance_score,
                "context_count": len(test_case.context),
            },
        )


class InformationDensityMetric(BaseMetric):
    """Evaluates how information-dense and concise the answer is"""

    def __init__(self, threshold: float = 3.0):
        super().__init__(threshold)

    def evaluate(self, test_case: TestCase) -> EvaluationResult:
        """
        Stub implementation for information density evaluation.
        In a real implementation, this would use an LLM to assess if the answer
        provides sufficient information without being overly verbose or too brief.
        """
        # Stub logic: analyze answer length, content richness, and relevance
        question_lower = test_case.question.lower()
        answer_lower = test_case.answer.lower()
        context_text = " ".join(test_case.context).lower()

        # Calculate various density factors
        answer_length = len(test_case.answer.split())
        question_length = len(test_case.question.split())

        # Information richness: unique informative words in answer
        answer_words = set(answer_lower.split())
        context_words = set(context_text.split())
        question_words = set(question_lower.split())

        # Count informative words (words that add value beyond common words)
        # fmt: off
        common_words = { "the", "a", "an", "and", "or", "but", "in", "on", "at",
        "to", "for", "of", "with", "by", "is", "are", "was", "were", "be",
        "been", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "can", }
        # fmt: on
        informative_words = answer_words - common_words

        # Relevance density: how many informative words are relevant to the question
        relevant_informative = informative_words.intersection(
            question_words.union(context_words)
        )

        # Calculate density scores
        if answer_length == 0:
            density_score = 0.0
        else:
            # Information density: informative words per total words
            info_density = len(informative_words) / answer_length

            # Relevance density: relevant informative words per informative words
            relevance_density = len(relevant_informative) / max(
                len(informative_words), 1
            )

            # Length appropriateness: not too short, not too long
            expected_length = max(question_length * 2, 5)  # Rough heuristic
            length_appropriateness = 1.0 - min(
                abs(answer_length - expected_length) / expected_length, 1.0
            )

            # Combined density score
            density_score = (
                info_density * 0.4
                + relevance_density * 0.4
                + length_appropriateness * 0.2
            ) * 5

        reasoning = f"Info density: {len(informative_words)}/{answer_length} words, Relevance: {len(relevant_informative)}/{len(informative_words)} relevant, Length appropriateness: {length_appropriateness:.2f}"
        passed = self._determine_pass(density_score)

        return EvaluationResult(
            test_case_id=test_case.id,
            metric_name="information_density",
            score=self._score_to_enum(density_score),
            score_numeric=density_score,
            reasoning=reasoning,
            passed=passed,
            metadata={
                "answer_length": answer_length,
                "question_length": question_length,
                "informative_words": len(informative_words),
                "relevant_informative": len(relevant_informative),
                "length_appropriateness": length_appropriateness,
                "info_density": len(informative_words) / max(answer_length, 1),
                "relevance_density": len(relevant_informative)
                / max(len(informative_words), 1),
            },
        )


class Evaluator:
    """Main evaluation orchestrator."""

    def __init__(self):
        # Default metrics available
        self.default_metrics = {
            "answer_relevancy": AnswerRelevancyMetric(),
            "faithfulness": FaithfulnessMetric(),
            "contextual_precision": ContextualPrecisionMetric(),
            "information_density": InformationDensityMetric(),
        }

    def evaluate(
        self, test_cases: List[TestCase], metrics: Optional[List[BaseMetric]] = None
    ) -> TestRun:
        """
        Evaluate test cases using specified metrics.

        Args:
            test_cases: List of test cases to evaluate
            metrics: List of metric classes to use. If None, uses all default metrics.

        Returns:
            TestRun object with all results
        """
        if metrics is None:
            # Use all default metrics
            metrics = list(self.default_metrics.values())

        test_run = TestRun()

        for test_case in test_cases:
            test_run.add_test_case(test_case)

            for metric in metrics:
                try:
                    result = metric.evaluate(test_case)
                    test_run.add_result(result)
                except Exception as e:
                    # Create error result
                    error_result = EvaluationResult(
                        test_case_id=test_case.id,
                        metric_name=getattr(metric, "name", metric.__class__.__name__),
                        score=EvaluationScore.VERY_POOR,
                        score_numeric=0.0,
                        reasoning=f"Evaluation failed: {str(e)}",
                        passed=False,
                        metadata={"error": str(e)},
                    )
                    test_run.add_result(error_result)

        return test_run

    def add_metric(self, metric: BaseMetric):
        """Add a custom metric to the default metrics"""
        metric_name = getattr(metric, "name", metric.__class__.__name__.lower())
        self.default_metrics[metric_name] = metric


def evaluate(
    test_cases: List[TestCase], metrics: Optional[List[BaseMetric]] = None
) -> TestRun:
    """
    Convenience function to run the evaluation.

    Args:
        test_cases: List of test cases to evaluate
        metrics: List of metric classes to use

    Returns:
        TestRun object with all results
    """
    evaluator = Evaluator()
    return evaluator.evaluate(test_cases, metrics)


def create_sample_test_cases() -> List[TestCase]:
    """Create sample test cases for testing the evaluation framework"""
    return [
        TestCase(
            question="What is the capital of France?",
            answer="The capital of France is Paris, which is located in the northern part of the country.",
            context=[
                "Paris is the capital and most populous city of France.",
                "France is a country in Western Europe.",
                "The Eiffel Tower is a famous landmark in Paris.",
            ],
            ground_truth="Paris",
            metadata={"category": "geography", "difficulty": "easy"},
        ),
        TestCase(
            question="What is the largest planet in our solar system?",
            answer="Jupiter is the largest planet in our solar system.",
            context=[
                "Jupiter is the fifth planet from the Sun.",
                "Jupiter is the largest planet in our solar system.",
                "Saturn is the second largest planet.",
            ],
            ground_truth="Jupiter",
            metadata={"category": "astronomy", "difficulty": "easy"},
        ),
        TestCase(
            question="Who wrote Romeo and Juliet?",
            answer="William Shakespeare wrote Romeo and Juliet in the late 16th century.",
            context=[
                "Romeo and Juliet is a tragedy by William Shakespeare.",
                "The play was written between 1591 and 1596.",
                "It tells the story of two young lovers from feuding families.",
            ],
            ground_truth="William Shakespeare",
            metadata={"category": "literature", "difficulty": "medium"},
        ),
    ]


def main():
    """Example usage of the evaluation"""
    test_cases = create_sample_test_cases()

    print(
        "Using evaluate([test_cases], metrics=[answer_relevancy, faithfulness, contextual_precision, information_density])"
    )
    test_run = evaluate(
        test_cases,
        metrics=[
            AnswerRelevancyMetric(),
            FaithfulnessMetric(),
            ContextualPrecisionMetric(),
            InformationDensityMetric(),
        ],
    )

    # Display results
    print(f"Test Run ID: {test_run.id}")
    print(f"Total Test Cases: {len(test_run.test_cases)}")
    print(f"Total Results: {len(test_run.results)}")
    print()

    # Display individual results
    for test_case in test_run.test_cases:
        print(f"Test Case: {test_case.question}")
        print(f"Answer: {test_case.answer}")
        print(f"Context: {test_case.context}")

        # Get results for this test case
        case_results = [r for r in test_run.results if r.test_case_id == test_case.id]

        for result in case_results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            print(f"  {result.metric_name.replace('_', ' ').title()}: {status}")
            print(f"    Score: {result.score.name} ({result.score_numeric:.2f}/5.0)")
            print(f"    Reasoning: {result.reasoning}")

        print("-" * 50)

    # Display summary
    print("\nTest Run Summary:")
    print("=" * 20)
    summary = test_run.get_summary()

    for key, value in summary.items():
        if key == "metric_summary":
            print(f"{key}:")
            for metric, details in value.items():
                print(f"  {metric}: {details}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
