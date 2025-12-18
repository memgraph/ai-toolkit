"""Evaluation metrics for text quality assessment.

Note: This module requires optional dependencies.
Install with: pip install 'memgraph-toolbox[evaluations]'
"""


def __getattr__(name: str):
    """Lazy import to avoid loading heavy dependencies unless needed."""
    if name == "CoherenceEmbeddingsBasedMetric":
        from .coherence import CoherenceEmbeddingsBasedMetric

        return CoherenceEmbeddingsBasedMetric

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CoherenceEmbeddingsBasedMetric",
]
