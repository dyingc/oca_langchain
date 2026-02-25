"""
Tests for Response API Converter Functions

This module tests the converter functions in converters/responses_converter.py,
ensuring proper conversion between Response API format and LangChain messages.
"""

import pytest
import json
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage,
    SystemMessage,
)

from models.responses_types import (
    ResponseRequest,
    Response,
    ResponseStatus,
    FunctionTool,
    FunctionCall,
    FunctionCallOutput,
    EasyInputMessage,
    InputMessage,
    OutputMessage,
    OutputFunctionCall,
    OutputContentText,
    ResponseUsage,
)
from converters.responses_converter import (
    response_input_to_langchain_messages,
    response_tools_to_openai_tools,
    response_request_to_langchain,
    langchain_to_response_output,
    create_response_error,
    create_response_created_event,
    create_output_item_added_event,
    create_output_text_delta_event,
    create_function_call_arguments_delta_event,
    create_output_item_done_event,
    create_response_completed_event,
    format_stream_event,
)


class TestResponseInputToLangchain:
    """Test conversion of Response API input to LangChain messages."""

    def test_none_input(self):
        """Test handling of None input."""
        messages = response_input_to_langchain_messages(None)
        assert messages == []

    def test_string_input(self):
        """Test conversion of simple string input."""
        messages = response_input_to_langchain_messages("Hello, world!")
        assert len(messages) == 1
        assert isinstance(messages[0], HumanMessage)
        assert messages[0].content == "Hello, world!"

    def test_empty_string_input(self):
        """Test handling of empty string input."""
        messages = response_input_to_langchain_messages("")
        assert len(messages) == 1
        assert isinstance(messages[0], HumanMessage)
        assert messages[0].content == ""

    def test_instructions_are_added(self):
        """Test that instructions are added as system message."""
        messages = response_input_to_langchain_messages(
            "Hello",
            instructions="Be helpful"
        )
        assert len(messages) == 2
        assert isinstance(messages[0], SystemMessage)
        assert messages[0].content == "Be helpful"
        assert isinstance(messages[1], HumanMessage)

    def test_easy_input_message_dict(self):
        """Test conversion of EasyInputMessage (dict format)."""
        input_data = [{"role": "user", "content": "Hello", "type": "message"}]
        messages = response_input_to_langchain_messages(input_data)
        assert len(messages) == 1
        assert isinstance(messages[0], HumanMessage)
        assert messages[0].content == "Hello"

    def test_multiple_messages(self):
        """Test conversion of multiple messages."""
        input_data = [
            {"role": "user", "content": "Hello", "type": "message"},
            {"role": "assistant", "content": "Hi there!", "type": "message"},
            {"role": "user", "content": "How are you?", "type": "message"}
        ]
        messages = response_input_to_langchain_messages(input_data)
        assert len(messages) == 3
        assert isinstance(messages[0], HumanMessage)
        assert isinstance(messages[1], AIMessage)
        assert isinstance(messages[2], HumanMessage)

    def test_system_message(self):
        """Test conversion of system message."""
        input_data = [{"role": "system", "content": "You are helpful.", "type": "message"}]
        messages = response_input_to_langchain_messages(input_data)
        assert len(messages) == 1
        assert isinstance(messages[0], SystemMessage)

    def test_developer_message(self):
        """Test conversion of developer message (treated as system)."""
        input_data = [{"role": "developer", "content": "Code well.", "type": "message"}]
        messages = response_input_to_langchain_messages(input_data)
        assert len(messages) == 1
        assert isinstance(messages[0], SystemMessage)

    def test_function_call_dict(self):
        """Test conversion of function_call item."""
        input_data = [
            {"role": "user", "content": "What's the weather?", "type": "message"},
            {
                "type": "function_call",
                "call_id": "call_123",
                "name": "get_weather",
                "arguments": '{"city": "Tokyo"}'
            }
        ]
        messages = response_input_to_langchain_messages(input_data)
        assert len(messages) == 2
        assert isinstance(messages[1], AIMessage)
        assert messages[1].content == ""
        assert "tool_calls" in messages[1].additional_kwargs
        tool_calls = messages[1].additional_kwargs["tool_calls"]
        assert len(tool_calls) == 1
        assert tool_calls[0]["id"] == "call_123"
        assert tool_calls[0]["function"]["name"] == "get_weather"

    def test_function_call_output_dict(self):
        """Test conversion of function_call_output item."""
        input_data = [
            {
                "type": "function_call_output",
                "call_id": "call_123",
                "output": "Sunny, 25°C"
            }
        ]
        messages = response_input_to_langchain_messages(input_data)
        assert len(messages) == 1
        assert isinstance(messages[0], ToolMessage)
        assert messages[0].content == "Sunny, 25°C"
        assert messages[0].tool_call_id == "call_123"

    def test_function_call_output_dict_output(self):
        """Test function_call_output with dict output."""
        input_data = [
            {
                "type": "function_call_output",
                "call_id": "call_123",
                "output": {"temperature": 25, "condition": "sunny"}
            }
        ]
        messages = response_input_to_langchain_messages(input_data)
        assert len(messages) == 1
        assert isinstance(messages[0], ToolMessage)
        # Dict output should be JSON serialized
        output_data = json.loads(messages[0].content)
        assert output_data["temperature"] == 25

    def test_full_conversation_with_tools(self):
        """Test full conversation with tool calls."""
        input_data = [
            {"role": "user", "content": "What's the weather in Tokyo?", "type": "message"},
            {
                "type": "function_call",
                "call_id": "call_123",
                "name": "get_weather",
                "arguments": '{"city": "Tokyo"}'
            },
            {
                "type": "function_call_output",
                "call_id": "call_123",
                "output": "Sunny, 25°C"
            }
        ]
        messages = response_input_to_langchain_messages(input_data)
        assert len(messages) == 3
        assert isinstance(messages[0], HumanMessage)
        assert isinstance(messages[1], AIMessage)
        assert isinstance(messages[2], ToolMessage)


class TestResponseToolsConversion:
    """Test conversion of Response API tools to OpenAI format."""

    def test_none_tools(self):
        """Test handling of None tools."""
        result = response_tools_to_openai_tools(None)
        assert result is None

    def test_empty_list_tools(self):
        """Test handling of empty tool list."""
        result = response_tools_to_openai_tools([])
        assert result is None

    def test_function_tool_dict(self):
        """Test conversion of function tool (dict format)."""
        tools = [{
            "type": "function",
            "name": "get_weather",
            "description": "Get weather info",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                }
            }
        }]
        result = response_tools_to_openai_tools(tools)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "get_weather"
        assert result[0]["function"]["description"] == "Get weather info"

    def test_function_tool_pydantic(self):
        """Test conversion of function tool (Pydantic model)."""
        tools = [
            FunctionTool(
                name="get_weather",
                description="Get weather info",
                parameters={"type": "object", "properties": {"city": {"type": "string"}}}
            )
        ]
        result = response_tools_to_openai_tools(tools)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "get_weather"

    def test_builtin_tools_skipped(self):
        """Test that built-in tools are skipped."""
        tools = [
            {"type": "web_search"},
            {"type": "file_search", "vector_store_ids": ["vs_123"]}
        ]
        result = response_tools_to_openai_tools(tools)
        assert result is None

    def test_mixed_tools(self):
        """Test mixed function and built-in tools."""
        tools = [
            {"type": "function", "name": "test", "description": "Test", "parameters": {}},
            {"type": "web_search"}
        ]
        result = response_tools_to_openai_tools(tools)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "test"

    def test_custom_tool_type(self):
        """Test custom tool type (e.g., from Codex)."""
        tools = [{
            "type": "custom",
            "name": "apply_patch",
            "description": "Apply a patch to edit files",
            "parameters": {
                "type": "object",
                "properties": {
                    "file": {"type": "string"}
                }
            }
        }]
        result = response_tools_to_openai_tools(tools)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "apply_patch"

    def test_custom_tool_with_input_schema(self):
        """Test custom tool with input_schema instead of parameters."""
        tools = [{
            "type": "custom_tool",
            "name": "my_tool",
            "description": "My custom tool",
            "input_schema": {
                "type": "object",
                "properties": {
                    "arg": {"type": "string"}
                }
            }
        }]
        result = response_tools_to_openai_tools(tools)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "my_tool"
        # input_schema is converted to parameters and normalized with required fields
        params = result[0]["function"]["parameters"]
        assert params["type"] == "object"
        assert params["properties"] == {"arg": {"type": "string"}}
        # ensure_valid_parameters adds these if missing
        assert "required" in params
        assert "additionalProperties" in params

    def test_custom_tool_without_name(self):
        """Test custom tool without name (should be skipped)."""
        tools = [{
            "type": "custom",
            "description": "No name tool"
        }]
        result = response_tools_to_openai_tools(tools)
        # Should be skipped since no name
        assert result is None


class TestResponseRequestToLangchain:
    """Test full request conversion."""

    def test_minimal_request(self):
        """Test minimal request conversion."""
        request = ResponseRequest(model="gpt-4o")
        result = response_request_to_langchain(request)
        assert "messages" in result
        assert "tools" in result
        assert result["messages"] == []
        assert result["tools"] is None

    def test_full_request(self):
        """Test full request conversion."""
        request = ResponseRequest(
            model="gpt-4o",
            input="Hello",
            instructions="Be helpful",
            tools=[FunctionTool(name="test", description="Test", parameters={})],
            temperature=0.7,
            max_output_tokens=100
        )
        result = response_request_to_langchain(request)
        assert len(result["messages"]) == 2  # system + user
        assert isinstance(result["messages"][0], SystemMessage)
        assert isinstance(result["messages"][1], HumanMessage)
        assert result["tools"] is not None
        assert result["temperature"] == 0.7
        assert result["max_tokens"] == 100


class TestLangchainToResponseOutput:
    """Test conversion of LangChain output to Response API format."""

    def test_simple_text_response(self):
        """Test simple text response conversion."""
        lc_message = AIMessage(content="Hello, how can I help?")
        response = langchain_to_response_output(
            lc_message=lc_message,
            model="gpt-4o"
        )
        assert response.model == "gpt-4o"
        assert response.status == ResponseStatus.COMPLETED
        assert len(response.output) == 1
        assert response.output[0].type == "message"
        assert isinstance(response.output[0], OutputMessage)

    def test_response_with_tool_calls(self):
        """Test response with tool calls."""
        lc_message = AIMessage(
            content="Let me check the weather.",
            additional_kwargs={
                "tool_calls": [{
                    "type": "function",
                    "id": "call_123",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Tokyo"}'
                    }
                }]
            }
        )
        response = langchain_to_response_output(
            lc_message=lc_message,
            model="gpt-4o"
        )
        assert len(response.output) == 2  # message + function_call
        assert response.output[0].type == "message"
        assert response.output[1].type == "function_call"
        assert isinstance(response.output[1], OutputFunctionCall)
        assert response.output[1].name == "get_weather"

    def test_response_with_previous_id(self):
        """Test response with previous_response_id."""
        lc_message = AIMessage(content="Hello")
        response = langchain_to_response_output(
            lc_message=lc_message,
            model="gpt-4o",
            response_id="resp_456",
            previous_response_id="resp_123"
        )
        assert response.id == "resp_456"
        assert response.previous_response_id == "resp_123"

    def test_response_with_usage(self):
        """Test response with usage stats."""
        lc_message = AIMessage(content="Hello")
        response = langchain_to_response_output(
            lc_message=lc_message,
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50
        )
        assert response.usage.input_tokens == 100
        assert response.usage.output_tokens == 50
        assert response.usage.total_tokens == 150

    def test_empty_response(self):
        """Test empty AIMessage conversion."""
        lc_message = AIMessage(content="")
        response = langchain_to_response_output(
            lc_message=lc_message,
            model="gpt-4o"
        )
        assert response.status == ResponseStatus.COMPLETED
        assert len(response.output) == 1


class TestCreateResponseError:
    """Test error response creation."""

    def test_basic_error(self):
        """Test basic error response."""
        response = create_response_error(
            message="Something went wrong",
            error_type="server_error"
        )
        assert response.status == ResponseStatus.FAILED
        assert response.error["type"] == "server_error"
        assert response.error["message"] == "Something went wrong"

    def test_error_with_model(self):
        """Test error response with model."""
        response = create_response_error(
            message="Rate limit exceeded",
            error_type="rate_limit_error",
            model="gpt-4o"
        )
        assert response.model == "gpt-4o"


class TestStreamEventHelpers:
    """Test stream event helper functions."""

    def test_create_response_created_event(self):
        """Test response.created event creation."""
        event = create_response_created_event(
            response_id="resp_123",
            model="gpt-4o"
        )
        assert event["type"] == "response.created"
        assert event["response"]["id"] == "resp_123"
        assert event["response"]["model"] == "gpt-4o"
        assert event["response"]["status"] == "in_progress"

    def test_create_response_created_event_with_previous(self):
        """Test response.created event with previous_response_id."""
        event = create_response_created_event(
            response_id="resp_456",
            model="gpt-4o",
            previous_response_id="resp_123"
        )
        assert event["response"]["previous_response_id"] == "resp_123"

    def test_create_output_item_added_event_message(self):
        """Test output_item.added event for message."""
        event = create_output_item_added_event(
            output_index=0,
            item_type="message",
            item_id="msg_123"
        )
        assert event["type"] == "response.output_item.added"
        assert event["output_index"] == 0
        assert event["item"]["type"] == "message"
        assert event["item"]["role"] == "assistant"

    def test_create_output_item_added_event_function_call(self):
        """Test output_item.added event for function_call."""
        event = create_output_item_added_event(
            output_index=1,
            item_type="function_call",
            item_id="fc_123"
        )
        assert event["item"]["type"] == "function_call"
        assert event["item"]["status"] == "in_progress"

    def test_create_output_text_delta_event(self):
        """Test output_text.delta event creation."""
        event = create_output_text_delta_event(
            output_index=0,
            content_index=0,
            delta="Hello"
        )
        assert event["type"] == "response.output_text.delta"
        assert event["delta"] == "Hello"

    def test_create_function_call_arguments_delta_event(self):
        """Test function_call_arguments.delta event creation."""
        event = create_function_call_arguments_delta_event(
            output_index=1,
            call_id="call_123",
            delta='{"city'
        )
        assert event["type"] == "response.function_call_arguments.delta"
        assert event["call_id"] == "call_123"
        assert event["delta"] == '{"city'

    def test_create_output_item_done_event(self):
        """Test output_item.done event creation."""
        item = {
            "id": "msg_123",
            "type": "message",
            "content": [{"type": "output_text", "text": "Hello"}]
        }
        event = create_output_item_done_event(
            output_index=0,
            item=item
        )
        assert event["type"] == "response.output_item.done"
        assert event["item"]["id"] == "msg_123"

    def test_create_response_completed_event(self):
        """Test response.completed event creation."""
        event = create_response_completed_event(
            response_id="resp_123",
            model="gpt-4o",
            output=[{"type": "message", "content": []}],
            usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        )
        assert event["type"] == "response.completed"
        assert event["response"]["id"] == "resp_123"
        assert event["response"]["status"] == "completed"
        assert event["response"]["usage"]["total_tokens"] == 15

    def test_format_stream_event(self):
        """Test SSE formatting of stream events."""
        event = create_output_text_delta_event(
            output_index=0,
            content_index=0,
            delta="Hello"
        )
        formatted = format_stream_event(event)
        assert formatted.startswith("event: response.output_text.delta\n")
        assert "data:" in formatted
        assert formatted.endswith("\n\n")


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_unknown_input_format(self):
        """Test handling of unknown input format."""
        # Should convert to string
        messages = response_input_to_langchain_messages(12345)
        assert len(messages) == 1
        assert messages[0].content == "12345"

    def test_structured_content_in_message(self):
        """Test message with structured content blocks."""
        input_data = [{
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Hello"},
                {"type": "input_image", "image_url": "https://example.com/img.png"}
            ]
        }]
        messages = response_input_to_langchain_messages(input_data)
        assert len(messages) == 1
        # Should extract text from structured content
        assert "Hello" in messages[0].content

    def test_function_call_without_id(self):
        """Test function call without explicit ID (should generate one)."""
        input_data = [{
            "type": "function_call",
            "name": "test",
            "arguments": "{}"
        }]
        messages = response_input_to_langchain_messages(input_data)
        assert len(messages) == 1
        tool_calls = messages[0].additional_kwargs.get("tool_calls", [])
        assert len(tool_calls) == 1
        # ID should be generated
        assert tool_calls[0]["id"] is not None

    def test_multiple_tool_calls_in_single_response(self):
        """Test AIMessage with multiple tool calls."""
        lc_message = AIMessage(
            content="",
            additional_kwargs={
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "call_1",
                        "function": {"name": "func1", "arguments": "{}"}
                    },
                    {
                        "type": "function",
                        "id": "call_2",
                        "function": {"name": "func2", "arguments": "{}"}
                    }
                ]
            }
        )
        response = langchain_to_response_output(
            lc_message=lc_message,
            model="gpt-4o"
        )
        # Should have function call items
        function_calls = [item for item in response.output if item.type == "function_call"]
        assert len(function_calls) == 2
