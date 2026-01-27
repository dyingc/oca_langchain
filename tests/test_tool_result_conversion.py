#!/usr/bin/env python3
"""
Test tool_result message conversion from Anthropic to OpenAI format.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.anthropic_types import AnthropicMessage, AnthropicContentBlock
from converters.anthropic_request_converter import anthropic_to_langchain_messages
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


def test_simple_user_message():
    """Test simple user message conversion."""
    print("Test 1: Simple user message")
    messages = [
        AnthropicMessage(role="user", content="Hello")
    ]
    result = anthropic_to_langchain_messages(messages)

    assert len(result) == 1
    assert isinstance(result[0], HumanMessage)
    assert result[0].content == "Hello"
    print("  ✅ PASSED")


def test_user_message_with_content_blocks():
    """Test user message with text content blocks."""
    print("Test 2: User message with content blocks")
    messages = [
        AnthropicMessage(
            role="user",
            content=[
                AnthropicContentBlock(type="text", text="Hello"),
                AnthropicContentBlock(type="text", text="World")
            ]
        )
    ]
    result = anthropic_to_langchain_messages(messages)

    assert len(result) == 1
    assert isinstance(result[0], HumanMessage)
    assert "Hello" in result[0].content
    assert "World" in result[0].content
    print("  ✅ PASSED")


def test_tool_result_conversion():
    """Test tool_result block conversion to ToolMessage."""
    print("Test 3: Tool result conversion")
    messages = [
        AnthropicMessage(
            role="user",
            content=[
                AnthropicContentBlock(
                    type="tool_result",
                    tool_use_id="tool_1",
                    content="25°C, sunny"
                )
            ]
        )
    ]
    result = anthropic_to_langchain_messages(messages)

    assert len(result) == 1, f"Expected 1 message, got {len(result)}"
    assert isinstance(result[0], ToolMessage), f"Expected ToolMessage, got {type(result[0])}"
    assert result[0].content == "25°C, sunny"
    assert result[0].tool_call_id == "tool_1"
    print("  ✅ PASSED")


def test_mixed_content_with_tool_result():
    """Test message with both text and tool_result blocks."""
    print("Test 4: Mixed content with tool_result")
    messages = [
        AnthropicMessage(
            role="user",
            content=[
                AnthropicContentBlock(type="text", text="Here's the result:"),
                AnthropicContentBlock(
                    type="tool_result",
                    tool_use_id="tool_1",
                    content="25°C, sunny"
                )
            ]
        )
    ]
    result = anthropic_to_langchain_messages(messages)

    # Should create 2 messages: HumanMessage for text, ToolMessage for tool_result
    assert len(result) == 2, f"Expected 2 messages, got {len(result)}"
    assert isinstance(result[0], HumanMessage), f"First should be HumanMessage, got {type(result[0])}"
    assert isinstance(result[1], ToolMessage), f"Second should be ToolMessage, got {type(result[1])}"
    assert result[0].content == "Here's the result:"
    assert result[1].content == "25°C, sunny"
    assert result[1].tool_call_id == "tool_1"
    print("  ✅ PASSED")


def test_multiple_tool_results():
    """Test multiple tool_result blocks in one message."""
    print("Test 5: Multiple tool results")
    messages = [
        AnthropicMessage(
            role="user",
            content=[
                AnthropicContentBlock(
                    type="tool_result",
                    tool_use_id="tool_1",
                    content="Result 1"
                ),
                AnthropicContentBlock(
                    type="tool_result",
                    tool_use_id="tool_2",
                    content="Result 2"
                )
            ]
        )
    ]
    result = anthropic_to_langchain_messages(messages)

    assert len(result) == 2, f"Expected 2 messages, got {len(result)}"
    assert all(isinstance(m, ToolMessage) for m in result)
    assert result[0].tool_call_id == "tool_1"
    assert result[1].tool_call_id == "tool_2"
    print("  ✅ PASSED")


def test_full_tool_call_flow():
    """Test complete tool call flow: user → assistant (tool_use) → user (tool_result)."""
    print("Test 6: Full tool call flow")
    messages = [
        AnthropicMessage(role="user", content="What's the weather in Tokyo?"),
        AnthropicMessage(
            role="assistant",
            content=[
                AnthropicContentBlock(
                    type="tool_use",
                    id="tool_1",
                    name="get_weather",
                    input={"city": "Tokyo"}
                )
            ]
        ),
        AnthropicMessage(
            role="user",
            content=[
                AnthropicContentBlock(
                    type="tool_result",
                    tool_use_id="tool_1",
                    content="25°C, sunny"
                )
            ]
        )
    ]
    result = anthropic_to_langchain_messages(messages)

    assert len(result) == 3, f"Expected 3 messages, got {len(result)}"
    assert isinstance(result[0], HumanMessage)
    assert isinstance(result[1], AIMessage)
    assert isinstance(result[2], ToolMessage)

    # Check AIMessage has tool_calls
    assert "tool_calls" in result[1].additional_kwargs
    tool_calls = result[1].additional_kwargs["tool_calls"]
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "get_weather"

    # Check ToolMessage
    assert result[2].content == "25°C, sunny"
    assert result[2].tool_call_id == "tool_1"

    print("  ✅ PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("Tool Result Conversion Tests")
    print("=" * 60)

    try:
        test_simple_user_message()
        test_user_message_with_content_blocks()
        test_tool_result_conversion()
        test_mixed_content_with_tool_result()
        test_multiple_tool_results()
        test_full_tool_call_flow()

        print("\n" + "=" * 60)
        print("All tests PASSED! ✅")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n❌ Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
