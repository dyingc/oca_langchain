"""
End-to-end API tests for message validation functionality.

These tests make actual HTTP requests to the running API server
to verify that the message validation works correctly in production.

Requires a running API server (bash run_api.sh).
Run with: pytest tests/test_e2e_api_validation.py -m e2e
"""

import json

import pytest
import requests


API_BASE_URL = "http://127.0.0.1:8450"
MODEL_NAME = "oca/gpt-4.1"

pytestmark = pytest.mark.e2e


@pytest.fixture(autouse=True)
def check_server():
    """Skip all tests if API server is not running."""
    try:
        resp = requests.get(f"{API_BASE_URL}/v1/models", timeout=5)
        if resp.status_code != 200:
            pytest.skip("API server not responding correctly")
    except requests.exceptions.ConnectionError:
        pytest.skip("API server not running at " + API_BASE_URL)


class TestOpenAIEndToEnd:
    """E2E tests for OpenAI-compatible endpoints."""

    def test_valid_sequence(self):
        """Valid simple request should succeed."""
        response = requests.post(
            f"{API_BASE_URL}/v1/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "What is 1+1?"}],
                "stream": False,
            },
            timeout=30,
        )
        assert response.status_code == 200

    def test_interrupted_tool_calls_auto_fixed(self):
        """Interrupted tool_calls should be auto-fixed and succeed."""
        response = requests.post(
            f"{API_BASE_URL}/v1/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "type": "function",
                                "id": "call_interrupted_123",
                                "function": {
                                    "name": "web_search",
                                    "arguments": '{"query": "test"}',
                                },
                            }
                        ],
                    },
                    {"role": "user", "content": "Wait, stop that search!"},
                    {
                        "role": "tool",
                        "tool_call_id": "call_interrupted_123",
                        "content": '{"result": "This should be skipped"}',
                    },
                    {"role": "user", "content": "Just say hello"},
                ],
                "stream": False,
            },
            timeout=30,
        )
        assert response.status_code == 200

    def test_streaming(self):
        """Streaming response should work."""
        response = requests.post(
            f"{API_BASE_URL}/v1/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "Say 'hello world'"}],
                "stream": True,
            },
            stream=True,
            timeout=30,
        )
        assert response.status_code == 200

        chunk_count = 0
        for line in response.iter_lines():
            if line:
                chunk_count += 1
                if chunk_count >= 5:
                    break
        assert chunk_count > 0


class TestAnthropicEndToEnd:
    """E2E tests for Anthropic-compatible endpoints."""

    def test_valid_sequence(self):
        """Valid Anthropic request should succeed."""
        response = requests.post(
            f"{API_BASE_URL}/v1/messages",
            json={
                "model": MODEL_NAME,
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "What is 1+1?"}],
                "stream": False,
            },
            headers={"anthropic-version": "2023-06-01"},
            timeout=30,
        )
        assert response.status_code == 200

    def test_interrupted_tool_use_auto_fixed(self):
        """Interrupted tool_use should be auto-fixed and succeed."""
        response = requests.post(
            f"{API_BASE_URL}/v1/messages",
            json={
                "model": MODEL_NAME,
                "max_tokens": 100,
                "messages": [
                    {"role": "user", "content": "Search for something"},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_interrupted_123",
                                "name": "web_search",
                                "input": {"query": "test"},
                            }
                        ],
                    },
                    {"role": "user", "content": "Wait, stop!"},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_interrupted_123",
                                "content": '{"result": "skipped"}',
                            }
                        ],
                    },
                    {"role": "user", "content": "Just say hello"},
                ],
                "stream": False,
            },
            headers={"anthropic-version": "2023-06-01"},
            timeout=30,
        )
        assert response.status_code == 200
