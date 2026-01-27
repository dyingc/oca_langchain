#!/usr/bin/env python3
"""
Anthropic Messages API Usage Examples

This file demonstrates how to use the Anthropic-compatible /v1/messages endpoint
with both raw HTTP requests and the official Anthropic Python SDK.
"""

import anthropic
import requests
import json


# Configuration
BASE_URL = "http://127.0.0.1:8450"
API_KEY = "test"  # Optional, for validation only
MODEL_NAME = "oca/gpt-4.1"  # Adjust to your available model


def example_1_basic_message_http():
    """
    Example 1: Basic message using raw HTTP request
    """
    print("\n" + "="*60)
    print("Example 1: Basic Message (HTTP)")
    print("="*60)

    url = f"{BASE_URL}/v1/messages"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": API_KEY,
        "anthropic-version": "2023-06-01"
    }
    data = {
        "model": MODEL_NAME,
        "max_tokens": 100,
        "messages": [
            {
                "role": "user",
                "content": "Hello! Please respond with a short greeting."
            }
        ]
    }

    response = requests.post(url, headers=headers, json=data)
    print(f"Status: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")


def example_2_basic_message_sdk():
    """
    Example 2: Basic message using Anthropic Python SDK
    """
    print("\n" + "="*60)
    print("Example 2: Basic Message (Anthropic SDK)")
    print("="*60)

    client = anthropic.Anthropic(
        base_url=BASE_URL,
        api_key=API_KEY
    )

    message = client.messages.create(
        model=MODEL_NAME,
        max_tokens=100,
        messages=[
            {
                "role": "user",
                "content": "Hello! Please respond with a short greeting."
            }
        ]
    )

    print(f"Message ID: {message.id}")
    print(f"Model: {message.model}")
    print(f"Stop Reason: {message.stop_reason}")
    print(f"Content:\n{message.content[0].text}")


def example_3_multi_turn_conversation():
    """
    Example 3: Multi-turn conversation
    """
    print("\n" + "="*60)
    print("Example 3: Multi-turn Conversation")
    print("="*60)

    client = anthropic.Anthropic(
        base_url=BASE_URL,
        api_key=API_KEY
    )

    message = client.messages.create(
        model=MODEL_NAME,
        max_tokens=100,
        messages=[
            {
                "role": "user",
                "content": "My name is Alice"
            },
            {
                "role": "assistant",
                "content": "Nice to meet you, Alice!"
            },
            {
                "role": "user",
                "content": "What's my name?"
            }
        ]
    )

    print(f"Response:\n{message.content[0].text}")


def example_4_tool_use():
    """
    Example 4: Tool / Function calling
    """
    print("\n" + "="*60)
    print("Example 4: Tool Use")
    print("="*60)

    client = anthropic.Anthropic(
        base_url=BASE_URL,
        api_key=API_KEY
    )

    message = client.messages.create(
        model=MODEL_NAME,
        max_tokens=200,
        messages=[
            {
                "role": "user",
                "content": "What's the weather in Tokyo?"
            }
        ],
        tools=[
            {
                "name": "get_weather",
                "description": "Get the current weather in a location",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        }
                    },
                    "required": ["location"]
                }
            }
        ]
    )

    print(f"Stop Reason: {message.stop_reason}")
    print(f"Content Blocks:")
    for block in message.content:
        if block.type == "text":
            print(f"  [text]: {block.text}")
        elif block.type == "tool_use":
            print(f"  [tool_use] id={block.id}, name={block.name}")
            print(f"    input: {block.input}")


def example_5_streaming():
    """
    Example 5: Streaming response
    """
    print("\n" + "="*60)
    print("Example 5: Streaming Response")
    print("="*60)

    client = anthropic.Anthropic(
        base_url=BASE_URL,
        api_key=API_KEY
    )

    print("Streaming response:")
    with client.messages.stream(
        model=MODEL_NAME,
        max_tokens=100,
        messages=[
            {
                "role": "user",
                "content": "Count from 1 to 5"
            }
        ]
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
        print()  # New line after completion


def example_6_multipart_content():
    """
    Example 6: Multipart content (text + tool result)
    """
    print("\n" + "="*60)
    print("Example 6: Multipart Content (Tool Result)")
    print("="*60)

    client = anthropic.Anthropic(
        base_url=BASE_URL,
        api_key=API_KEY
    )

    # Simulate a tool use scenario
    message = client.messages.create(
        model=MODEL_NAME,
        max_tokens=200,
        messages=[
            {
                "role": "user",
                "content": "What's the weather in Tokyo?"
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "I'll check the weather in Tokyo for you."
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_123",
                        "name": "get_weather",
                        "input": {"location": "Tokyo"}
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_123",
                        "content": "The weather in Tokyo is 22°C and sunny."
                    }
                ]
            }
        ]
    )

    print(f"Response:\n{message.content[0].text}")


def example_7_error_handling():
    """
    Example 7: Error handling
    """
    print("\n" + "="*60)
    print("Example 7: Error Handling")
    print("="*60)

    client = anthropic.Anthropic(
        base_url=BASE_URL,
        api_key=API_KEY
    )

    # Test 1: Missing max_tokens
    try:
        message = client.messages.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": "Hello"
                }
            ]
        )
    except anthropic.APIError as e:
        print(f"✓ Caught expected error for missing max_tokens:")
        print(f"  {e}")

    # Test 2: Invalid model
    try:
        message = client.messages.create(
            model="invalid-model",
            max_tokens=100,
            messages=[
                {
                    "role": "user",
                    "content": "Hello"
                }
            ]
        )
    except anthropic.APIError as e:
        print(f"\n✓ Caught expected error for invalid model:")
        print(f"  {e}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Anthropic Messages API Examples")
    print("="*60)
    print(f"Base URL: {BASE_URL}")
    print(f"Model: {MODEL_NAME}")

    # Run all examples
    try:
        example_1_basic_message_http()
        example_2_basic_message_sdk()
        example_3_multi_turn_conversation()
        example_4_tool_use()
        example_5_streaming()
        example_6_multipart_content()
        example_7_error_handling()

        print("\n" + "="*60)
        print("All examples completed!")
        print("="*60)

    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        import traceback
        traceback.print_exc()
