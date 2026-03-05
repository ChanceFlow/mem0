from unittest.mock import Mock, patch

import pytest

from mem0.configs.embeddings.base import BaseEmbedderConfig
from mem0.embeddings.volcengine import VolcengineEmbedding


@pytest.fixture
def mock_volcengine_embedding_client() -> Mock:
    """Create a mocked Ark client for embeddings."""
    with patch("mem0.embeddings.volcengine.Ark") as mock_ark:
        mock_client: Mock = Mock()
        mock_ark.return_value = mock_client
        yield mock_client


def test_embed_default_model(mock_volcengine_embedding_client: Mock) -> None:
    """Verify embedding call with default model."""
    config: BaseEmbedderConfig = BaseEmbedderConfig()
    embedder: VolcengineEmbedding = VolcengineEmbedding(config)
    mock_response: Mock = Mock()
    mock_response.data = {"embedding": [0.1, 0.2, 0.3], "object": "embedding"}
    mock_volcengine_embedding_client.multimodal_embeddings.create.return_value = mock_response

    result: list[float] = embedder.embed("Hello world")

    mock_volcengine_embedding_client.multimodal_embeddings.create.assert_called_once_with(
        model="doubao-embedding-text-240715",
        encoding_format="float",
        input=[{"text": "Hello world", "type": "text"}],
    )
    assert result == [0.1, 0.2, 0.3]


def test_embed_custom_model(mock_volcengine_embedding_client: Mock) -> None:
    """Verify embedding call with custom model."""
    config: BaseEmbedderConfig = BaseEmbedderConfig(model="doubao-embedding-vision-250615")
    embedder: VolcengineEmbedding = VolcengineEmbedding(config)
    mock_response: Mock = Mock()
    mock_response.data = {"embedding": [0.4, 0.5, 0.6], "object": "embedding"}
    mock_volcengine_embedding_client.multimodal_embeddings.create.return_value = mock_response

    result: list[float] = embedder.embed("Test embedding")

    mock_volcengine_embedding_client.multimodal_embeddings.create.assert_called_once_with(
        model="doubao-embedding-vision-250615",
        encoding_format="float",
        input=[{"text": "Test embedding", "type": "text"}],
    )
    assert result == [0.4, 0.5, 0.6]


def test_embed_removes_newlines(mock_volcengine_embedding_client: Mock) -> None:
    """Verify newline normalization before embedding request."""
    config: BaseEmbedderConfig = BaseEmbedderConfig()
    embedder: VolcengineEmbedding = VolcengineEmbedding(config)
    mock_response: Mock = Mock()
    mock_response.data = {"embedding": [0.7, 0.8, 0.9], "object": "embedding"}
    mock_volcengine_embedding_client.multimodal_embeddings.create.return_value = mock_response

    result: list[float] = embedder.embed("Hello\nworld")

    mock_volcengine_embedding_client.multimodal_embeddings.create.assert_called_once_with(
        model="doubao-embedding-text-240715",
        encoding_format="float",
        input=[{"text": "Hello world", "type": "text"}],
    )
    assert result == [0.7, 0.8, 0.9]


def test_embed_handles_sdk_object_payload(mock_volcengine_embedding_client: Mock) -> None:
    """Verify embedding parsing when SDK returns object-style data.embedding."""
    config: BaseEmbedderConfig = BaseEmbedderConfig()
    embedder: VolcengineEmbedding = VolcengineEmbedding(config)
    mock_data: Mock = Mock()
    mock_data.embedding = [1.1, 1.2, 1.3]
    mock_response: Mock = Mock()
    mock_response.data = mock_data
    mock_volcengine_embedding_client.multimodal_embeddings.create.return_value = mock_response

    result: list[float] = embedder.embed("Object payload")

    assert result == [1.1, 1.2, 1.3]
