from sentence_transformers import SentenceTransformer
import numpy as np

from ..utils.embedding import get_sentence_transformer_model, get_model_device


def test_get_sentence_transformer_model_default():
    """Test the get_sentence_transformer_model function with default arguments."""

    # Test with default arguments
    model = get_sentence_transformer_model()
    # Verify the model is a SentenceTransformer instance
    assert isinstance(model, SentenceTransformer)

    # Test that we can get the device
    device = get_model_device(model)
    assert device is not None
    assert isinstance(device, str)

    # Test that the model can encode text
    sample_texts = ["Hello world", "Graph databases are fast"]
    embeddings = model.encode(sample_texts, convert_to_tensor=False)

    # Verify embeddings are generated
    assert embeddings is not None
    assert len(embeddings) == 2  # Two input texts
    assert len(embeddings[0]) > 0  # Each embedding should have dimensions

    # Verify embeddings are numpy arrays (default behavior)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == 2  # Two input texts
    assert embeddings.shape[1] > 0  # Each embedding should have dimensions
    assert embeddings.dtype == np.float32


def test_get_sentence_transformer_model_custom_device():
    """Test the get_sentence_transformer_model function with custom device."""

    # Test with CPU device explicitly
    model = get_sentence_transformer_model(device="cpu")
    # Verify the model is a SentenceTransformer instance
    assert isinstance(model, SentenceTransformer)

    # Verify device is CPU
    device = get_model_device(model)
    assert "cpu" in device.lower()


def test_get_sentence_transformer_model_data_parallel():
    """Test the get_sentence_transformer_model function with DataParallel option."""

    # Test with DataParallel (will only work if CUDA is available)
    model = get_sentence_transformer_model(use_data_parallel=True)

    # Verify the model is still a SentenceTransformer instance (not DataParallel)
    assert isinstance(model, SentenceTransformer)

    # Test that the model can still encode text
    sample_texts = ["Test text for DataParallel"]
    embeddings = model.encode(sample_texts, convert_to_tensor=False)

    # Verify embeddings are generated
    assert embeddings is not None
    assert len(embeddings) == 1
    assert len(embeddings[0]) > 0
