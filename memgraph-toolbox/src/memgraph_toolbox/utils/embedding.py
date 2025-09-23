from typing import List, Optional

import torch
from sentence_transformers import SentenceTransformer


# NOTE: HF_TOKEN has to be set in the environment variables.
def get_sentence_transformer_model(
    name: str = "all-MiniLM-L6-v2",
    device: Optional[str] = None,
    use_data_parallel: bool = False,
    gpu_ids: Optional[List[int]] = None,
) -> SentenceTransformer:
    """
    Get a sentence transformer model with optional DataParallel support.
    Args:
        name: Model name or path
        device: Specific device to use ('cpu', 'cuda', 'mps', or None for auto-detection)
        use_data_parallel: Whether to wrap model with DataParallel for multi-GPU
        gpu_ids: List of GPU IDs to use for DataParallel (default: all available)

    Returns:
        SentenceTransformer model. Always returns the underlying model, not DataParallel
        wrapper because then it's possible to use the whole SentenceTransformer API.
    """
    # Auto-detect device if not specified
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    model = SentenceTransformer(name, device=device)
    # Wrap with DataParallel if requested and CUDA is available
    if use_data_parallel and torch.cuda.is_available():
        if gpu_ids is None:
            gpu_ids = list(range(torch.cuda.device_count()))
        # Create DataParallel wrapper but return the underlying model
        parallel_model = torch.nn.DataParallel(model, device_ids=gpu_ids)
        # Return the underlying model to avoid DataParallel wrapper issues
        return parallel_model.module
    return model


def get_model_device(model: SentenceTransformer) -> str:
    """Get the device where the model is located."""
    return str(next(model.parameters()).device)


if __name__ == "__main__":
    print("Available PyTorch backends:")
    print(f"  CUDA: {torch.cuda.is_available()}")
    print(
        f"  MPS: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}"
    )
    print(f"  CPU: True (always available)")

    # Example usage
    print("\nExample usage:")
    model = get_sentence_transformer_model()
    print(f"Model device: {get_model_device(model)}")
    print(f"Model type: {type(model).__name__}")

    # Example of using the model directly (no wrapper needed)
    print("\nDirect usage example:")
    sample_texts = ["Hello world", "Graph databases are fast"]
    print(f"Sample texts: {sample_texts}")
    print("Use model.encode() directly with all SentenceTransformer options:")
    print("  - model.encode(texts, convert_to_tensor=True)")
    print("  - model.encode(texts, batch_size=32)")
    print("  - model.encode(texts, normalize_embeddings=True)")
    print("  - model.encode(texts, show_progress_bar=True)")

    # Test with DataParallel (if multiple GPUs available)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print("\nTesting DataParallel:")
        parallel_model = get_sentence_transformer_model(use_data_parallel=True)
        print(f"Parallel model device: {get_model_device(parallel_model)}")
        print(f"Model type: {type(parallel_model).__name__}")
        print(
            "Note: DataParallel is used internally but returns clean SentenceTransformer"
        )
