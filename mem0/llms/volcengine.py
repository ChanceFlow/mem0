import json
import os
from typing import Any, Dict, List, Optional, Union

from volcenginesdkarkruntime import Ark

from mem0.configs.llms.base import BaseLlmConfig
from mem0.configs.llms.volcengine import VolcengineConfig
from mem0.llms.base import LLMBase
from mem0.memory.utils import extract_json


class VolcengineLLM(LLMBase):
    """LLM provider implementation for Volcengine Ark chat completions."""

    def __init__(self, config: Optional[Union[BaseLlmConfig, VolcengineConfig, Dict[str, Any]]] = None) -> None:
        """
        Initialize Volcengine Ark client.

        Args:
            config: Provider configuration object or dictionary.
        """
        if config is None:
            normalized_config: VolcengineConfig = VolcengineConfig()
        elif isinstance(config, dict):
            normalized_config = VolcengineConfig(**config)
        elif isinstance(config, BaseLlmConfig) and not isinstance(config, VolcengineConfig):
            normalized_config = VolcengineConfig(
                model=config.model,
                temperature=config.temperature,
                api_key=config.api_key,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                top_k=config.top_k,
                enable_vision=config.enable_vision,
                vision_details=config.vision_details,
                http_client_proxies=config.http_client,
            )
        else:
            normalized_config = config

        super().__init__(normalized_config)

        if not self.config.model:
            self.config.model = "doubao-seed-2-0-pro-250415"

        api_key: Optional[str] = self.config.api_key or os.getenv("ARK_API_KEY")
        base_url: str = (
            self.config.ark_base_url or os.getenv("ARK_BASE_URL") or "https://ark.cn-beijing.volces.com/api/v3"
        )
        self.client: Ark = Ark(api_key=api_key, base_url=base_url)

    def _parse_response(self, response: Any, tools: Optional[List[Dict[str, Any]]]) -> Union[str, Dict[str, Any]]:
        """
        Parse a chat completion response into mem0-compatible structure.

        Args:
            response: Raw provider response object.
            tools: Tool definitions used in the request.

        Returns:
            Parsed text content or dict including tool calls.
        """
        message: Any = response.choices[0].message
        if tools:
            processed_response: Dict[str, Any] = {"content": message.content, "tool_calls": []}
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    processed_response["tool_calls"].append(
                        {
                            "name": tool_call.function.name,
                            "arguments": json.loads(extract_json(tool_call.function.arguments)),
                        }
                    )
            return processed_response
        return message.content

    def _should_retry_without_response_format(self, error: Exception, response_format: Optional[Any]) -> bool:
        """
        Decide whether to retry without `response_format`.

        Args:
            error: Raised exception from Ark client request.
            response_format: Requested response format.

        Returns:
            True if the request should be retried without response_format.
        """
        if not response_format:
            return False

        if not isinstance(response_format, dict):
            return False

        response_format_type: Optional[str] = response_format.get("type")
        if response_format_type not in {"json_object", "json_schema"}:
            return False

        error_message: str = str(error).lower()
        unsupported_markers: List[str] = [
            "response_format.type",
            "json_object is not supported",
            "json_schema is not supported",
            "not support response_format",
            "invalidparameter",
        ]
        return any(marker in error_message for marker in unsupported_markers)

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Any] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
        **kwargs: Any,
    ) -> Union[str, Dict[str, Any]]:
        """
        Generate a response using Volcengine Ark chat completions.

        Args:
            messages: Conversation messages.
            response_format: Optional structured response format.
            tools: Optional tool definitions.
            tool_choice: Tool invocation policy.
            **kwargs: Provider-specific request arguments.

        Returns:
            Plain text response or tool-call response payload.
        """
        params: Dict[str, Any] = self._get_supported_params(messages=messages, **kwargs)
        params.update(
            {
                "model": self.config.model,
                "messages": messages,
            }
        )

        if response_format:
            params["response_format"] = response_format
        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice

        try:
            response: Any = self.client.chat.completions.create(**params)
        except Exception as error:
            if self._should_retry_without_response_format(error=error, response_format=response_format):
                fallback_params: Dict[str, Any] = dict(params)
                fallback_params.pop("response_format", None)
                response = self.client.chat.completions.create(**fallback_params)
            else:
                raise
        return self._parse_response(response, tools)
