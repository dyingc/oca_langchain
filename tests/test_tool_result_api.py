#!/usr/bin/env python3
"""
Test tool_result API request.
"""

import httpx
import json

BASE_URL = "http://127.0.0.1:8450/v1/messages"
HEADERS = {
    "Content-Type": "application/json",
    "anthropic-version": "2023-06-01"
}

payload = {
    "model": "oca/gpt-4.1",
    "max_tokens": 100,
    "messages": [
        {"role": "user", "content": "What is the weather in Tokyo?"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "tool_1", "name": "get_weather", "input": {"city": "Tokyo"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tool_1", "content": "25 degrees Celsius, sunny"}]}
    ]
}

print("Testing tool_result API request...")
print(f"Payload: {json.dumps(payload, indent=2)}")
print()

try:
    response = httpx.post(BASE_URL, json=payload, headers=HEADERS, timeout=30.0)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text[:500]}")
except Exception as e:
    print(f"Error: {e}")
