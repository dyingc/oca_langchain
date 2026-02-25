"""
Integration Tests for Response API Endpoints

This module tests the Response API endpoints with mocked model responses,
ensuring proper request handling and response generation.

Note: These tests mock the chat model to avoid requiring actual API credentials.
"""

import pytest
import json
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from fastapi import FastAPI

from langchain_core.messages import AIMessage

from responses_api import (
    create_response,
    get_response,
    delete_response,
    store_response,
    get_stored_response,
    delete_stored_response,
    _response_store,
)
from models.responses_types import (
    ResponseRequest,
    Response,
    ResponseStatus,
    OutputMessage,
    OutputFunctionCall,
    OutputContentText,
)


# Create a test FastAPI app
@pytest.fixture
def app():
    """Create a test FastAPI app with Response API routes."""
    from fastapi import FastAPI
    app = FastAPI()

    @app.post("/v1/responses")
    async def responses_create(request: ResponseRequest):
        return await create_response(request)

    @app.get("/v1/responses/{response_id}")
    async def responses_get(response_id: str):
        return await get_response(response_id)

    @app.delete("/v1/responses/{response_id}")
    async def responses_delete(response_id: str):
        return await delete_response(response_id)

    return app


@pytest.fixture
def client(app):
    """Create a test client."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def clear_stores():
    """Clear response stores before each test."""
    _response_store.clear()
    yield
    _response_store.clear()


@pytest.fixture
def mock_chat_model():
    """Create a mock chat model."""
    mock = MagicMock()
    mock.available_models = ["oca/gpt-4o", "oca/gpt-4o-mini", "oca/gpt-5.2"]
    mock.model = "oca/gpt-4o"
    mock.temperature = 0.7

    # Mock invoke to return a simple AIMessage
    mock.invoke = MagicMock(return_value=AIMessage(content="Hello! How can I help you today?"))

    # Mock astream as async generator
    async def mock_astream(*args, **kwargs):
        from langchain_core.messages import AIMessageChunk
        from langchain_core.outputs import ChatGenerationChunk
        chunks = ["Hello", "! ", "How ", "can ", "I ", "help ", "you", "?"]
        for chunk in chunks:
            yield ChatGenerationChunk(message=AIMessageChunk(content=chunk))

    mock.astream = mock_astream

    return mock


class TestCreateResponseEndpoint:
    """Test POST /v1/responses endpoint."""

    @patch("responses_api._get_chat_model")
    def test_create_response_minimal(self, mock_get_model, client, mock_chat_model):
        """Test minimal request to create response."""
        mock_get_model.return_value = mock_chat_model

        response = client.post(
            "/v1/responses",
            json={"model": "oca/gpt-4o"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "oca/gpt-4o"
        assert data["object"] == "response"
        assert data["status"] == "completed"

    @patch("responses_api._get_chat_model")
    def test_create_response_with_input(self, mock_get_model, client, mock_chat_model):
        """Test request with input."""
        mock_get_model.return_value = mock_chat_model

        response = client.post(
            "/v1/responses",
            json={
                "model": "oca/gpt-4o",
                "input": "What is the weather?"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "oca/gpt-4o"

    @patch("responses_api._get_chat_model")
    def test_create_response_with_instructions(self, mock_get_model, client, mock_chat_model):
        """Test request with instructions."""
        mock_get_model.return_value = mock_chat_model

        response = client.post(
            "/v1/responses",
            json={
                "model": "oca/gpt-4o",
                "input": "Hello",
                "instructions": "Be helpful and concise."
            }
        )

        assert response.status_code == 200

    @patch("responses_api._get_chat_model")
    def test_create_response_with_message_input(self, mock_get_model, client, mock_chat_model):
        """Test request with message-style input."""
        mock_get_model.return_value = mock_chat_model

        response = client.post(
            "/v1/responses",
            json={
                "model": "oca/gpt-4o",
                "input": [
                    {"role": "user", "content": "Hello", "type": "message"}
                ]
            }
        )

        assert response.status_code == 200

    @patch("responses_api._get_chat_model")
    def test_create_response_with_tools(self, mock_get_model, client, mock_chat_model):
        """Test request with tools."""
        mock_get_model.return_value = mock_chat_model

        response = client.post(
            "/v1/responses",
            json={
                "model": "oca/gpt-4o",
                "input": "What's the weather?",
                "tools": [
                    {
                        "type": "function",
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string"}
                            }
                        }
                    }
                ]
            }
        )

        assert response.status_code == 200

    def test_create_response_missing_model(self, client):
        """Test request without model."""
        response = client.post(
            "/v1/responses",
            json={}
        )

        # Should return validation error
        assert response.status_code == 422

    @patch("responses_api._get_chat_model")
    def test_create_response_invalid_model(self, mock_get_model, client, mock_chat_model):
        """Test request with invalid model (has oca/ prefix but doesn't exist)."""
        mock_get_model.return_value = mock_chat_model

        response = client.post(
            "/v1/responses",
            json={"model": "oca/invalid-model"}
        )

        assert response.status_code == 404
        data = response.json()
        assert "not_found_error" in str(data["detail"])

    @patch("responses_api._get_chat_model")
    def test_create_response_model_resolution(self, mock_get_model, client, mock_chat_model):
        """Test that model without oca/ prefix gets resolved to default."""
        mock_get_model.return_value = mock_chat_model

        response = client.post(
            "/v1/responses",
            json={"model": "gpt-4o"}  # No oca/ prefix, should resolve to env default
        )

        # Should succeed because it falls back to default model
        assert response.status_code == 200
        data = response.json()
        # Response should use the resolved model (oca/gpt-5.2 from env)
        assert "oca/" in data["model"]

    @patch("responses_api._get_chat_model")
    def test_create_response_stored(self, mock_get_model, client, mock_chat_model):
        """Test that response is stored when store=True."""
        mock_get_model.return_value = mock_chat_model

        response = client.post(
            "/v1/responses",
            json={
                "model": "oca/gpt-4o",
                "input": "Hello",
                "store": True
            }
        )

        assert response.status_code == 200
        data = response.json()

        # Verify response is stored
        stored = get_stored_response(data["id"])
        assert stored is not None
        assert stored.model == "oca/gpt-4o"


class TestStreamingResponse:
    """Test streaming responses."""

    @patch("responses_api._get_chat_model")
    def test_streaming_response(self, mock_get_model, client, mock_chat_model):
        """Test streaming response."""
        mock_get_model.return_value = mock_chat_model

        response = client.post(
            "/v1/responses",
            json={
                "model": "oca/gpt-4o",
                "input": "Hello",
                "stream": True
            }
        )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")

        # Parse SSE events
        content = response.text
        assert "event: response.created" in content
        assert "event: response.completed" in content

    @patch("responses_api._get_chat_model")
    def test_streaming_text_deltas(self, mock_get_model, client, mock_chat_model):
        """Test that streaming sends text deltas."""
        mock_get_model.return_value = mock_chat_model

        response = client.post(
            "/v1/responses",
            json={
                "model": "oca/gpt-4o",
                "input": "Hello",
                "stream": True
            }
        )

        content = response.text
        # Should have text delta events
        assert "response.output_text.delta" in content


class TestGetResponseEndpoint:
    """Test GET /v1/responses/{id} endpoint."""

    def test_get_response_not_found(self, client):
        """Test getting non-existent response."""
        response = client.get("/v1/responses/nonexistent")
        assert response.status_code == 404

    def test_get_response_exists(self, client):
        """Test getting existing response."""
        # Store a response
        stored_response = Response(
            id="resp_test123",
            model="oca/gpt-4o",
            output=[OutputMessage(content=[OutputContentText(text="Hello")])]
        )
        store_response(stored_response)

        response = client.get("/v1/responses/resp_test123")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "resp_test123"
        assert data["model"] == "oca/gpt-4o"


class TestDeleteResponseEndpoint:
    """Test DELETE /v1/responses/{id} endpoint."""

    def test_delete_response_not_found(self, client):
        """Test deleting non-existent response."""
        response = client.delete("/v1/responses/nonexistent")
        assert response.status_code == 404

    def test_delete_response_exists(self, client):
        """Test deleting existing response."""
        # Store a response
        stored_response = Response(
            id="resp_test123",
            model="oca/gpt-4o"
        )
        store_response(stored_response)

        response = client.delete("/v1/responses/resp_test123")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "resp_test123"
        assert data["deleted"] is True

        # Verify it's deleted
        assert get_stored_response("resp_test123") is None


class TestStatefulConversation:
    """Test stateful conversation features."""

    @patch("responses_api._get_chat_model")
    def test_previous_response_id_continuation(self, mock_get_model, client, mock_chat_model):
        """Test conversation continuation with previous_response_id."""
        mock_get_model.return_value = mock_chat_model

        # Create first response
        response1 = client.post(
            "/v1/responses",
            json={
                "model": "oca/gpt-4o",
                "input": "Hello",
                "store": True
            }
        )
        assert response1.status_code == 200
        data1 = response1.json()

        # Create second response with previous_response_id
        response2 = client.post(
            "/v1/responses",
            json={
                "model": "oca/gpt-4o",
                "input": "How are you?",
                "store": True,
                "previous_response_id": data1["id"]
            }
        )
        assert response2.status_code == 200
        data2 = response2.json()
        assert data2["previous_response_id"] == data1["id"]

    @patch("responses_api._get_chat_model")
    def test_invalid_previous_response_id(self, mock_get_model, client, mock_chat_model):
        """Test error when previous_response_id doesn't exist."""
        mock_get_model.return_value = mock_chat_model

        response = client.post(
            "/v1/responses",
            json={
                "model": "oca/gpt-4o",
                "input": "Hello",
                "previous_response_id": "nonexistent"
            }
        )
        assert response.status_code == 404


class TestToolCallsInResponse:
    """Test tool calls in responses."""

    @patch("responses_api._get_chat_model")
    def test_response_with_tool_calls(self, mock_get_model, client):
        """Test response that includes tool calls."""
        # Create mock that returns tool calls
        mock_model = MagicMock()
        mock_model.available_models = ["oca/gpt-4o"]
        mock_model.model = "oca/gpt-4o"
        mock_model.invoke = MagicMock(return_value=AIMessage(
            content="Let me check that.",
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
        ))
        mock_get_model.return_value = mock_model

        response = client.post(
            "/v1/responses",
            json={
                "model": "oca/gpt-4o",
                "input": "What's the weather in Tokyo?"
            }
        )

        assert response.status_code == 200
        data = response.json()
        # Should have function_call output item
        output_types = [item["type"] for item in data["output"]]
        assert "function_call" in output_types


class TestResponseStore:
    """Test response storage functions."""

    def test_store_and_retrieve(self):
        """Test storing and retrieving a response."""
        response = Response(
            id="resp_test",
            model="oca/gpt-4o"
        )
        store_response(response)

        retrieved = get_stored_response("resp_test")
        assert retrieved is not None
        assert retrieved.id == "resp_test"
        assert retrieved.model == "oca/gpt-4o"

    def test_delete_response(self):
        """Test deleting a stored response."""
        response = Response(
            id="resp_test",
            model="oca/gpt-4o"
        )
        store_response(response)

        result = delete_stored_response("resp_test")
        assert result is True

        retrieved = get_stored_response("resp_test")
        assert retrieved is None

    def test_delete_nonexistent(self):
        """Test deleting non-existent response."""
        result = delete_stored_response("nonexistent")
        assert result is False


class TestEdgeCases:
    """Test edge cases and error handling."""

    @patch("responses_api._get_chat_model")
    def test_empty_input(self, mock_get_model, client, mock_chat_model):
        """Test request with empty input."""
        mock_get_model.return_value = mock_chat_model

        response = client.post(
            "/v1/responses",
            json={
                "model": "oca/gpt-4o",
                "input": ""
            }
        )

        # Should still work
        assert response.status_code == 200

    @patch("responses_api._get_chat_model")
    def test_complex_input_sequence(self, mock_get_model, client, mock_chat_model):
        """Test complex input with multiple item types."""
        mock_get_model.return_value = mock_chat_model

        response = client.post(
            "/v1/responses",
            json={
                "model": "oca/gpt-4o",
                "input": [
                    {"role": "user", "content": "What's the weather?", "type": "message"},
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
                    },
                    {"role": "user", "content": "Thanks!", "type": "message"}
                ]
            }
        )

        assert response.status_code == 200

    @patch("responses_api._get_chat_model")
    def test_model_error_handling(self, mock_get_model, client):
        """Test handling of model errors."""
        mock_model = MagicMock()
        mock_model.available_models = ["oca/gpt-4o"]
        mock_model.invoke = MagicMock(side_effect=Exception("API Error"))
        mock_get_model.return_value = mock_model

        response = client.post(
            "/v1/responses",
            json={
                "model": "oca/gpt-4o",
                "input": "Hello"
            }
        )

        assert response.status_code == 500
        data = response.json()
        assert "error" in data["detail"]

    @patch("responses_api._get_chat_model")
    def test_response_with_metadata(self, mock_get_model, client, mock_chat_model):
        """Test request with metadata."""
        mock_get_model.return_value = mock_chat_model

        response = client.post(
            "/v1/responses",
            json={
                "model": "oca/gpt-4o",
                "input": "Hello",
                "metadata": {
                    "user_id": "user_123",
                    "session_id": "session_456"
                }
            }
        )

        assert response.status_code == 200
