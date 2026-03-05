import os
from typing import Any, Dict, List, Literal, Optional, cast

from volcenginesdkarkruntime import Ark

from mem0.configs.embeddings.base import BaseEmbedderConfig
from mem0.embeddings.base import EmbeddingBase


class VolcengineEmbedding(EmbeddingBase):
    """Embedding provider implementation for Volcengine Ark."""

    def __init__(self, config: Optional[BaseEmbedderConfig] = None) -> None:
        """
        Initialize Ark embedding client.

        Args:
            config: Embedding configuration.
        """
        super().__init__(config)
        self.config.model = self.config.model or "doubao-embedding-text-240715"

        api_key: Optional[str] = self.config.api_key or os.getenv("ARK_API_KEY")
        ark_base_url: str = (
            getattr(self.config, "ark_base_url", None)
            or os.getenv("ARK_BASE_URL")
            or "https://ark.cn-beijing.volces.com/api/v3"
        )
        self.client: Ark = Ark(api_key=api_key, base_url=ark_base_url)

    def embed(self, text: str, memory_action: Optional[Literal["add", "search", "update"]] = None) -> List[float]:
        """
        Get embeddings for text using Ark multimodal embeddings in text-only mode.

        Args:
            text: Input text to embed.
            memory_action: Memory action type (unused by this provider).

        Returns:
            Embedding vector.
        """
        del memory_action
        sanitized_text: str = text.replace("\n", " ")
        response: Any = self.client.multimodal_embeddings.create(
            model=self.config.model,
            encoding_format="float",
            input=[{"text": sanitized_text, "type": "text"}],
        )

        data: Any = response.data
        if hasattr(data, "embedding"):
            return cast(List[float], getattr(data, "embedding"))
        if isinstance(data, dict):
            return cast(List[float], data.get("embedding", []))
        if isinstance(data, list) and data:
            first_item: Any = data[0]
            if isinstance(first_item, dict):
                return cast(List[float], first_item.get("embedding", []))
            if hasattr(first_item, "embedding"):
                return cast(List[float], getattr(first_item, "embedding"))
        return []
