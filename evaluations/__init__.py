"""
Knowledge Retrieval Evaluation Tools

This package provides functionality for evaluating knowledge retrieval
responses in different scenarios. The aim is to design tools to be LLM agnostic
and easily extensible.
"""

from .main import (
    TestCase,
    TestRun,
    EvaluationResult,
    EvaluationScore,
    BaseMetric,
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    InformationDensityMetric,
    Evaluator,
    evaluate,
    print_evaluation_summary,
)

__version__ = "0.1.0"

__all__ = [
    "TestCase",
    "TestRun",
    "EvaluationResult",
    "EvaluationScore",
    "BaseMetric",
    "AnswerRelevancyMetric",
    "FaithfulnessMetric",
    "ContextualPrecisionMetric",
    "InformationDensityMetric",
    "Evaluator",
    "evaluate",
    "print_evaluation_summary",
]
