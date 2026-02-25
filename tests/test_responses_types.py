"""
Tests for Response API Pydantic Models

This module tests the Pydantic models defined in models/responses_types.py,
ensuring proper validation, serialization, and deserialization.
"""

import pytest
import json
from pydantic import ValidationError

from models.responses_types import (
    # Request/Response
    ResponseRequest,
    Response,
    ResponseStatus,
    ResponseUsage,
    ResponseDeleted,
    # Input items
    EasyInputMessage,
    InputMessage,
    FunctionCall,
    FunctionCallOutput,
    InputContentText,
    InputContentImage,
    # Output items
    OutputMessage,
    OutputFunctionCall,
    OutputContentText,
    OutputContentRefusal,
    # Tools
    FunctionTool,
    WebSearchTool,
    FileSearchTool,
    ComputerUseTool,
    # Stream events
    ResponseCreatedEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputTextDeltaEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseOutputItemDoneEvent,
    ResponseCompletedEvent,
    # Helpers
    generate_response_id,
    generate_item_id,
)


class TestIDGeneration:
    """Test ID generation helpers."""

    def test_generate_response_id_format(self):
        """Test that response ID has correct format."""
        response_id = generate_response_id()
        assert response_id.startswith("resp_")
        assert len(response_id) == 29  # "resp_" + 24 chars

    def test_generate_response_id_uniqueness(self):
        """Test that response IDs are unique."""
        ids = set()
        for _ in range(100):
            response_id = generate_response_id()
            assert response_id not in ids
            ids.add(response_id)

    def test_generate_item_id_format(self):
        """Test that item ID has correct format."""
        item_id = generate_item_id("msg")
        assert item_id.startswith("msg_")
        assert len(item_id) == 28  # "msg_" + 24 chars

    def test_generate_item_id_different_prefixes(self):
        """Test item ID with different prefixes."""
        for prefix in ["msg", "fc", "rs", "call"]:
            item_id = generate_item_id(prefix)
            assert item_id.startswith(f"{prefix}_")


class TestInputContent:
    """Test input content types."""

    def test_input_content_text(self):
        """Test InputContentText model."""
        content = InputContentText(text="Hello, world!")
        assert content.type == "input_text"
        assert content.text == "Hello, world!"

    def test_input_content_image_url(self):
        """Test InputContentImage with URL."""
        content = InputContentImage(image_url="https://example.com/image.png")
        assert content.type == "input_image"
        assert content.image_url == "https://example.com/image.png"
        assert content.detail == "auto"

    def test_input_content_image_file_id(self):
        """Test InputContentImage with file ID."""
        content = InputContentImage(file_id="file-123")
        assert content.type == "input_image"
        assert content.file_id == "file-123"


class TestInputItems:
    """Test input item types."""

    def test_easy_input_message(self):
        """Test EasyInputMessage model."""
        msg = EasyInputMessage(role="user", content="Hello")
        assert msg.type == "message"
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_easy_input_message_roles(self):
        """Test all valid roles for EasyInputMessage."""
        for role in ["user", "assistant", "system", "developer"]:
            msg = EasyInputMessage(role=role, content="Test")
            assert msg.role == role

    def test_input_message_string_content(self):
        """Test InputMessage with string content."""
        msg = InputMessage(role="user", content="Hello")
        assert msg.type == "message"
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_input_message_structured_content(self):
        """Test InputMessage with structured content."""
        msg = InputMessage(
            role="user",
            content=[
                InputContentText(text="Hello"),
                InputContentImage(image_url="https://example.com/img.png")
            ]
        )
        assert msg.type == "message"
        assert isinstance(msg.content, list)
        assert len(msg.content) == 2

    def test_function_call(self):
        """Test FunctionCall model."""
        fc = FunctionCall(
            type="function_call",
            call_id="call_123",
            name="get_weather",
            arguments='{"city": "Tokyo"}'
        )
        assert fc.type == "function_call"
        assert fc.call_id == "call_123"
        assert fc.name == "get_weather"
        assert fc.arguments == '{"city": "Tokyo"}'

    def test_function_call_output(self):
        """Test FunctionCallOutput model."""
        fco = FunctionCallOutput(
            type="function_call_output",
            call_id="call_123",
            output="Sunny, 25°C"
        )
        assert fco.type == "function_call_output"
        assert fco.call_id == "call_123"
        assert fco.output == "Sunny, 25°C"


class TestOutputItems:
    """Test output item types."""

    def test_output_content_text(self):
        """Test OutputContentText model."""
        content = OutputContentText(text="Hello")
        assert content.type == "output_text"
        assert content.text == "Hello"
        assert content.annotations is None

    def test_output_content_text_with_annotations(self):
        """Test OutputContentText with annotations."""
        content = OutputContentText(
            text="Hello",
            annotations=[{"type": "citation", "url": "https://example.com"}]
        )
        assert content.annotations is not None
        assert len(content.annotations) == 1

    def test_output_content_refusal(self):
        """Test OutputContentRefusal model."""
        content = OutputContentRefusal(refusal="I cannot help with that.")
        assert content.type == "refusal"
        assert content.refusal == "I cannot help with that."

    def test_output_message(self):
        """Test OutputMessage model."""
        msg = OutputMessage(
            content=[OutputContentText(text="Hello")]
        )
        assert msg.type == "message"
        assert msg.role == "assistant"
        assert msg.status == "completed"
        assert len(msg.content) == 1

    def test_output_function_call(self):
        """Test OutputFunctionCall model."""
        fc = OutputFunctionCall(
            call_id="call_123",
            name="get_weather",
            arguments='{"city": "Tokyo"}'
        )
        assert fc.type == "function_call"
        assert fc.call_id == "call_123"
        assert fc.name == "get_weather"
        assert fc.arguments == '{"city": "Tokyo"}'
        assert fc.status == "completed"


class TestTools:
    """Test tool types."""

    def test_function_tool(self):
        """Test FunctionTool model."""
        tool = FunctionTool(
            type="function",
            name="get_weather",
            description="Get weather information",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                }
            }
        )
        assert tool.type == "function"
        assert tool.name == "get_weather"
        assert tool.description == "Get weather information"
        assert "properties" in tool.parameters

    def test_web_search_tool(self):
        """Test WebSearchTool model."""
        tool = WebSearchTool()
        assert tool.type == "web_search"
        assert tool.search_context_size == "medium"

    def test_file_search_tool(self):
        """Test FileSearchTool model."""
        tool = FileSearchTool(vector_store_ids=["vs_123"])
        assert tool.type == "file_search"
        assert tool.vector_store_ids == ["vs_123"]

    def test_computer_use_tool(self):
        """Test ComputerUseTool model."""
        tool = ComputerUseTool()
        assert tool.type == "computer"
        assert tool.environment == "browser"


class TestResponseRequest:
    """Test ResponseRequest model."""

    def test_minimal_request(self):
        """Test minimal valid request."""
        request = ResponseRequest(model="gpt-4o")
        assert request.model == "gpt-4o"
        assert request.input is None
        assert request.stream is False
        assert request.store is True

    def test_request_with_string_input(self):
        """Test request with string input."""
        request = ResponseRequest(
            model="gpt-4o",
            input="What is the weather?"
        )
        assert request.input == "What is the weather?"

    def test_request_with_list_input(self):
        """Test request with list input."""
        request = ResponseRequest(
            model="gpt-4o",
            input=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ]
        )
        assert isinstance(request.input, list)
        assert len(request.input) == 2

    def test_request_with_tools(self):
        """Test request with tools."""
        request = ResponseRequest(
            model="gpt-4o",
            tools=[
                FunctionTool(
                    name="get_weather",
                    description="Get weather",
                    parameters={"type": "object"}
                )
            ]
        )
        assert request.tools is not None
        assert len(request.tools) == 1

    def test_request_with_all_options(self):
        """Test request with all options."""
        request = ResponseRequest(
            model="gpt-4o",
            input="Hello",
            instructions="Be helpful",
            temperature=0.7,
            max_output_tokens=1000,
            top_p=0.9,
            stream=True,
            store=False,
            previous_response_id="resp_123",
            metadata={"user_id": "user_123"}
        )
        assert request.model == "gpt-4o"
        assert request.instructions == "Be helpful"
        assert request.temperature == 0.7
        assert request.max_output_tokens == 1000
        assert request.stream is True
        assert request.store is False
        assert request.previous_response_id == "resp_123"
        assert request.metadata["user_id"] == "user_123"


class TestResponse:
    """Test Response model."""

    def test_minimal_response(self):
        """Test minimal valid response."""
        response = Response(model="gpt-4o")
        assert response.model == "gpt-4o"
        assert response.object == "response"
        assert response.status == ResponseStatus.COMPLETED
        assert response.output == []

    def test_response_with_output(self):
        """Test response with output items."""
        response = Response(
            model="gpt-4o",
            output=[
                OutputMessage(
                    content=[OutputContentText(text="Hello!")]
                )
            ]
        )
        assert len(response.output) == 1
        assert response.output[0].type == "message"

    def test_response_with_function_call(self):
        """Test response with function call."""
        response = Response(
            model="gpt-4o",
            output=[
                OutputFunctionCall(
                    call_id="call_123",
                    name="get_weather",
                    arguments='{"city": "Tokyo"}'
                )
            ]
        )
        assert len(response.output) == 1
        assert response.output[0].type == "function_call"

    def test_response_with_usage(self):
        """Test response with usage stats."""
        response = Response(
            model="gpt-4o",
            usage=ResponseUsage(
                input_tokens=100,
                output_tokens=50,
                total_tokens=150
            )
        )
        assert response.usage.input_tokens == 100
        assert response.usage.output_tokens == 50
        assert response.usage.total_tokens == 150

    def test_response_failed(self):
        """Test failed response."""
        response = Response(
            model="gpt-4o",
            status=ResponseStatus.FAILED,
            error={
                "type": "server_error",
                "message": "Something went wrong"
            }
        )
        assert response.status == ResponseStatus.FAILED
        assert response.error["type"] == "server_error"

    def test_response_serialization(self):
        """Test response JSON serialization."""
        response = Response(
            id="resp_123",
            model="gpt-4o",
            output=[
                OutputMessage(
                    id="msg_456",
                    content=[OutputContentText(text="Hello!")]
                )
            ]
        )
        json_str = response.model_dump_json()
        data = json.loads(json_str)
        assert data["id"] == "resp_123"
        assert data["model"] == "gpt-4o"
        assert data["object"] == "response"


class TestStreamEvents:
    """Test streaming event types."""

    def test_response_created_event(self):
        """Test ResponseCreatedEvent."""
        response = Response(model="gpt-4o")
        event = ResponseCreatedEvent(response=response)
        assert event.type == "response.created"
        assert event.response.model == "gpt-4o"

    def test_output_item_added_event(self):
        """Test ResponseOutputItemAddedEvent."""
        item = OutputMessage(content=[])
        event = ResponseOutputItemAddedEvent(
            output_index=0,
            item=item
        )
        assert event.type == "response.output_item.added"
        assert event.output_index == 0

    def test_output_text_delta_event(self):
        """Test ResponseOutputTextDeltaEvent."""
        event = ResponseOutputTextDeltaEvent(
            output_index=0,
            content_index=0,
            delta="Hello"
        )
        assert event.type == "response.output_text.delta"
        assert event.delta == "Hello"

    def test_function_call_arguments_delta_event(self):
        """Test ResponseFunctionCallArgumentsDeltaEvent."""
        event = ResponseFunctionCallArgumentsDeltaEvent(
            output_index=0,
            call_id="call_123",
            delta='{"city'
        )
        assert event.type == "response.function_call_arguments.delta"
        assert event.call_id == "call_123"
        assert event.delta == '{"city'

    def test_output_item_done_event(self):
        """Test ResponseOutputItemDoneEvent."""
        item = OutputMessage(
            id="msg_123",
            content=[OutputContentText(text="Hello")]
        )
        event = ResponseOutputItemDoneEvent(
            output_index=0,
            item=item
        )
        assert event.type == "response.output_item.done"

    def test_response_completed_event(self):
        """Test ResponseCompletedEvent."""
        response = Response(
            model="gpt-4o",
            output=[OutputMessage(content=[OutputContentText(text="Hello")])],
            usage=ResponseUsage(input_tokens=10, output_tokens=5, total_tokens=15)
        )
        event = ResponseCompletedEvent(response=response)
        assert event.type == "response.completed"
        assert event.response.usage.total_tokens == 15


class TestResponseDeleted:
    """Test ResponseDeleted model."""

    def test_response_deleted(self):
        """Test ResponseDeleted model."""
        deleted = ResponseDeleted(id="resp_123")
        assert deleted.id == "resp_123"
        assert deleted.object == "response.deleted"
        assert deleted.deleted is True


class TestModelValidation:
    """Test model validation."""

    def test_invalid_role_validation(self):
        """Test that invalid roles are rejected."""
        with pytest.raises(ValidationError):
            EasyInputMessage(role="invalid_role", content="test")

    def test_invalid_type_validation(self):
        """Test that invalid types are rejected."""
        with pytest.raises(ValidationError):
            FunctionCall(type="invalid_type", name="test", arguments="{}")

    def test_missing_required_fields(self):
        """Test that missing required fields raise errors."""
        with pytest.raises(ValidationError):
            FunctionCall()  # Missing name and arguments


class TestDictConversion:
    """Test conversion to/from dictionaries."""

    def test_request_from_dict(self):
        """Test creating Request from dict."""
        data = {
            "model": "gpt-4o",
            "input": "Hello",
            "stream": True
        }
        request = ResponseRequest(**data)
        assert request.model == "gpt-4o"
        assert request.input == "Hello"
        assert request.stream is True

    def test_request_with_dict_input(self):
        """Test request with dict-style input items."""
        data = {
            "model": "gpt-4o",
            "input": [
                {"role": "user", "content": "Hello", "type": "message"},
                {"type": "function_call", "call_id": "call_123", "name": "test", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "call_123", "output": "result"}
            ]
        }
        request = ResponseRequest(**data)
        assert isinstance(request.input, list)
        assert len(request.input) == 3

    def test_response_to_dict(self):
        """Test converting Response to dict."""
        response = Response(
            id="resp_123",
            model="gpt-4o",
            output=[OutputMessage(content=[OutputContentText(text="Hello")])]
        )
        data = response.model_dump()
        assert data["id"] == "resp_123"
        assert data["model"] == "gpt-4o"
        assert len(data["output"]) == 1

    def test_response_exclude_none(self):
        """Test excluding None values from dict."""
        response = Response(model="gpt-4o")
        data = response.model_dump(exclude_none=True)
        assert "previous_response_id" not in data
        assert "error" not in data
