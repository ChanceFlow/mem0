import os
from unittest.mock import Mock, patch

import pytest

from mem0.configs.llms.base import BaseLlmConfig
from mem0.configs.llms.volcengine import VolcengineConfig
from mem0.llms.volcengine import VolcengineLLM


@pytest.fixture
def mock_volcengine_client() -> Mock:
    """Create a mocked Ark client instance."""
    with patch("mem0.llms.volcengine.Ark") as mock_ark:
        mock_client: Mock = Mock()
        mock_ark.return_value = mock_client
        yield mock_client


def test_volcengine_llm_base_url() -> None:
    """Verify base URL resolution order for Volcengine LLM."""
    with patch("mem0.llms.volcengine.Ark") as mock_ark:
        VolcengineLLM(BaseLlmConfig(model="doubao-seed", api_key="api_key"))
        mock_ark.assert_called_with(api_key="api_key", base_url="https://ark.cn-beijing.volces.com/api/v3")

    with patch("mem0.llms.volcengine.Ark") as mock_ark:
        provider_base_url: str = "https://api.provider.com/v3/"
        os.environ["ARK_BASE_URL"] = provider_base_url
        VolcengineLLM(VolcengineConfig(model="doubao-seed", api_key="api_key"))
        mock_ark.assert_called_with(api_key="api_key", base_url=provider_base_url)
        del os.environ["ARK_BASE_URL"]

    with patch("mem0.llms.volcengine.Ark") as mock_ark:
        config_base_url: str = "https://api.config.com/v3/"
        VolcengineLLM(
            VolcengineConfig(
                model="doubao-seed",
                api_key="api_key",
                ark_base_url=config_base_url,
            )
        )
        mock_ark.assert_called_with(api_key="api_key", base_url=config_base_url)


def test_generate_response_without_tools(mock_volcengine_client: Mock) -> None:
    """Verify plain-text response path."""
    llm: VolcengineLLM = VolcengineLLM(
        BaseLlmConfig(model="doubao-seed", temperature=0.7, max_tokens=100, top_p=1.0)
    )
    messages: list[dict[str, str]] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
    ]
    mock_response: Mock = Mock()
    mock_response.choices = [Mock(message=Mock(content="I'm doing well, thank you for asking!"))]
    mock_volcengine_client.chat.completions.create.return_value = mock_response

    response: str = llm.generate_response(messages)

    mock_volcengine_client.chat.completions.create.assert_called_once_with(
        model="doubao-seed",
        messages=messages,
        temperature=0.7,
        max_tokens=100,
        top_p=1.0,
    )
    assert response == "I'm doing well, thank you for asking!"


def test_generate_response_with_tools(mock_volcengine_client: Mock) -> None:
    """Verify tool call parsing path."""
    llm: VolcengineLLM = VolcengineLLM(
        BaseLlmConfig(model="doubao-seed", temperature=0.7, max_tokens=100, top_p=1.0)
    )
    messages: list[dict[str, str]] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Add a new memory: Today is a sunny day."},
    ]
    tools: list[dict[str, object]] = [
        {
            "type": "function",
            "function": {
                "name": "add_memory",
                "description": "Add a memory",
                "parameters": {
                    "type": "object",
                    "properties": {"data": {"type": "string", "description": "Data to add to memory"}},
                    "required": ["data"],
                },
            },
        }
    ]

    mock_response: Mock = Mock()
    mock_message: Mock = Mock()
    mock_message.content = "I've added the memory for you."
    mock_tool_call: Mock = Mock()
    mock_tool_call.function.name = "add_memory"
    mock_tool_call.function.arguments = '{"data": "Today is a sunny day."}'
    mock_message.tool_calls = [mock_tool_call]
    mock_response.choices = [Mock(message=mock_message)]
    mock_volcengine_client.chat.completions.create.return_value = mock_response

    response: dict[str, object] = llm.generate_response(messages, tools=tools)

    mock_volcengine_client.chat.completions.create.assert_called_once_with(
        model="doubao-seed",
        messages=messages,
        temperature=0.7,
        max_tokens=100,
        top_p=1.0,
        tools=tools,
        tool_choice="auto",
    )
    assert response["content"] == "I've added the memory for you."
    assert len(response["tool_calls"]) == 1
    assert response["tool_calls"][0]["name"] == "add_memory"
    assert response["tool_calls"][0]["arguments"] == {"data": "Today is a sunny day."}
