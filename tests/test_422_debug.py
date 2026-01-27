#!/usr/bin/env python3
"""
Debug script to test Anthropic SDK with our local API
and capture any 422 errors.
"""

import httpx
import json
import sys

# Test 1: Direct HTTP request to see raw response
def test_raw_http():
    print("=" * 60)
    print("Test 1: Raw HTTP request (streaming)")
    print("=" * 60)

    url = "http://127.0.0.1:8450/v1/messages"
    headers = {
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01"
    }
    payload = {
        "model": "oca/gpt-4.1",
        "max_tokens": 50,
        "messages": [{"role": "user", "content": "Say hello"}],
        "stream": True
    }

    try:
        with httpx.stream("POST", url, json=payload, headers=headers, timeout=30.0) as response:
            print(f"Status: {response.status_code}")
            print(f"Headers: {dict(response.headers)}")
            print("\nSSE Events:")
            for line in response.iter_lines():
                if line:
                    print(f"  {line}")
    except Exception as e:
        print(f"Error: {e}")

# Test 2: Use Anthropic SDK
def test_anthropic_sdk():
    print("\n" + "=" * 60)
    print("Test 2: Anthropic SDK (streaming)")
    print("=" * 60)

    try:
        from anthropic import Anthropic

        client = Anthropic(
            api_key="test-key",  # Not used by our API
            base_url="http://127.0.0.1:8450"
        )

        print("Sending streaming request...")
        with client.messages.stream(
            model="oca/gpt-4.1",
            max_tokens=50,
            messages=[{"role": "user", "content": "Say hello"}]
        ) as stream:
            print("Response text: ", end="")
            for text in stream.text_stream:
                print(text, end="")
            print()

        print("SUCCESS: No 422 error!")

    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

# Test 3: Non-streaming with SDK
def test_anthropic_sdk_non_streaming():
    print("\n" + "=" * 60)
    print("Test 3: Anthropic SDK (non-streaming)")
    print("=" * 60)

    try:
        from anthropic import Anthropic

        client = Anthropic(
            api_key="test-key",
            base_url="http://127.0.0.1:8450"
        )

        print("Sending non-streaming request...")
        response = client.messages.create(
            model="oca/gpt-4.1",
            max_tokens=50,
            messages=[{"role": "user", "content": "Say hello"}]
        )

        print(f"Response: {response.content[0].text}")
        print("SUCCESS: No error!")

    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_raw_http()
    test_anthropic_sdk()
    test_anthropic_sdk_non_streaming()
