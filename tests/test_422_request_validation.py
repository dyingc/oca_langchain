#!/usr/bin/env python3
"""
Test various request formats to find what causes 422 error.
"""

import httpx
import json

BASE_URL = "http://127.0.0.1:8450/v1/messages"
HEADERS = {
    "Content-Type": "application/json",
    "anthropic-version": "2023-06-01"
}


def test_request(name: str, payload: dict):
    """Test a request and print the result."""
    print(f"\n{'='*60}")
    print(f"Test: {name}")
    print(f"{'='*60}")
    print(f"Payload: {json.dumps(payload, indent=2)[:500]}")

    try:
        response = httpx.post(BASE_URL, json=payload, headers=HEADERS, timeout=30.0)
        print(f"Status: {response.status_code}")
        if response.status_code == 422:
            print(f"422 Error Details: {response.text}")
        elif response.status_code == 200:
            print(f"Success! Response preview: {response.text[:200]}...")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Test 1: Valid request
    test_request("Valid request", {
        "model": "oca/gpt-4.1",
        "max_tokens": 50,
        "messages": [{"role": "user", "content": "Hi"}]
    })

    # Test 2: Missing max_tokens
    test_request("Missing max_tokens", {
        "model": "oca/gpt-4.1",
        "messages": [{"role": "user", "content": "Hi"}]
    })

    # Test 3: Invalid message role
    test_request("Invalid message role", {
        "model": "oca/gpt-4.1",
        "max_tokens": 50,
        "messages": [{"role": "invalid_role", "content": "Hi"}]
    })

    # Test 4: Empty messages
    test_request("Empty messages array", {
        "model": "oca/gpt-4.1",
        "max_tokens": 50,
        "messages": []
    })

    # Test 5: Content as array (Anthropic format)
    test_request("Content as array", {
        "model": "oca/gpt-4.1",
        "max_tokens": 50,
        "messages": [{
            "role": "user",
            "content": [{"type": "text", "text": "Hi"}]
        }]
    })

    # Test 6: Missing content field
    test_request("Missing content in message", {
        "model": "oca/gpt-4.1",
        "max_tokens": 50,
        "messages": [{"role": "user"}]
    })

    # Test 7: Extra unknown fields
    test_request("Extra unknown fields", {
        "model": "oca/gpt-4.1",
        "max_tokens": 50,
        "messages": [{"role": "user", "content": "Hi"}],
        "unknown_field": "value"
    })

    # Test 8: Wrong type for max_tokens
    test_request("max_tokens as string", {
        "model": "oca/gpt-4.1",
        "max_tokens": "50",  # string instead of int
        "messages": [{"role": "user", "content": "Hi"}]
    })

    # Test 9: Null values
    test_request("Null temperature", {
        "model": "oca/gpt-4.1",
        "max_tokens": 50,
        "temperature": None,
        "messages": [{"role": "user", "content": "Hi"}]
    })

    # Test 10: Tool result format
    test_request("Tool result message", {
        "model": "oca/gpt-4.1",
        "max_tokens": 50,
        "messages": [
            {"role": "user", "content": "What's the weather?"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "tool_1", "name": "get_weather", "input": {"city": "Tokyo"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tool_1", "content": "25°C, sunny"}]}
        ]
    })
