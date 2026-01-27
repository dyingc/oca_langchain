"""
Anthropic Request/Response Converter

This module provides conversion functions between Anthropic's Messages API format
and LangChain's message format, enabling seamless integration with the OCAChatModel.
"""

import json
import uuid
from typing import List, Dict, Any, Optional, Union
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage,
    BaseMessage,
    SystemMessage
)

from models.anthropic_types import (
    AnthropicRequest,
    AnthropicMessage,
    AnthropicContentBlock,
    AnthropicResponse,
    AnthropicToolDefinition,
    AnthropicUsage,
    AnthropicErrorResponse,
)


def anthropic_to_langchain_messages(
    anthropic_messages: List[AnthropicMessage]
) -> List[BaseMessage]:
    """
    Convert Anthropic messages to LangChain messages.

    Args:
        anthropic_messages: List of AnthropicMessage objects

    Returns:
        List of LangChain BaseMessage objects

    Example:
        Anthropic:
        {"role": "user", "content": "Hello"}

        → LangChain:
        HumanMessage(content="Hello")
    """
    lc_messages: List[BaseMessage] = []

    for msg in anthropic_messages:
        if msg.role == "user":
            # Handle content: can be string or list of content blocks
            if isinstance(msg.content, str):
                content_str = msg.content
            elif isinstance(msg.content, list):
                # Extract text from content blocks
                text_parts = []
                for block in msg.content:
                    if block.type == "text":
                        text_parts.append(block.text or "")
                    elif block.type == "image":
                        # Future: handle image blocks
                        text_parts.append("[Image]")
                    elif block.type == "tool_result":
                        # Handle tool result
                        tool_content = block.content
                        if isinstance(tool_content, str):
                            text_parts.append(tool_content)
                        elif isinstance(tool_content, list):
                            # Extract text from tool result content blocks
                            for sub_block in tool_content:
                                if isinstance(sub_block, dict) and sub_block.get("type") == "text":
                                    text_parts.append(sub_block.get("text", ""))
                                elif isinstance(sub_block, str):
                                    text_parts.append(sub_block)
                content_str = "\n".join(text_parts)
            else:
                content_str = str(msg.content)

            lc_messages.append(HumanMessage(content=content_str))

        elif msg.role == "assistant":
            # Handle content: can be string or list of content blocks
            if isinstance(msg.content, str):
                content_str = msg.content
                tool_calls = None
            elif isinstance(msg.content, list):
                # Check for tool_use blocks
                tool_use_blocks = [b for b in msg.content if b.type == "tool_use"]
                text_blocks = [b for b in msg.content if b.type == "text"]

                # Extract text content
                content_str = "\n".join([b.text or "" for b in text_blocks])

                # Convert tool_use blocks to OpenAI format
                if tool_use_blocks:
                    tool_calls = []
                    for block in tool_use_blocks:
                        tool_call = {
                            "type": "function",
                            "id": block.id or f"toolu_{uuid.uuid4().hex[:24]}",
                            "function": {
                                "name": block.name,
                                "arguments": json.dumps(block.input) if isinstance(block.input, dict) else "{}"
                            }
                        }
                        tool_calls.append(tool_call)
                else:
                    tool_calls = None
            else:
                content_str = str(msg.content)
                tool_calls = None

            # Create AIMessage with tool_calls if present
            if tool_calls:
                lc_messages.append(
                    AIMessage(content=content_str, additional_kwargs={"tool_calls": tool_calls})
                )
            else:
                lc_messages.append(AIMessage(content=content_str))

        elif msg.role == "system":
            # Handle system messages (if supported in future)
            if isinstance(msg.content, str):
                content_str = msg.content
            else:
                content_str = str(msg.content)
            lc_messages.append(SystemMessage(content=content_str))

    return lc_messages


def anthropic_tools_to_openai_tools(
    anthropic_tools: Optional[List[AnthropicToolDefinition]]
) -> Optional[List[Dict[str, Any]]]:
    """
    Convert Anthropic tool definitions to OpenAI format.

    Args:
        anthropic_tools: List of AnthropicToolDefinition objects

    Returns:
        List of OpenAI-formatted tool definitions

    Example:
        Anthropic:
        {
            "name": "weather",
            "description": "Get weather info",
            "input_schema": {"type": "object", "properties": {...}}
        }

        → OpenAI:
        {
            "type": "function",
            "function": {
                "name": "weather",
                "description": "Get weather info",
                "parameters": {"type": "object", "properties": {...}}
            }
        }
    """
    if not anthropic_tools:
        return None

    openai_tools = []
    for tool in anthropic_tools:
        openai_tool = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema
            }
        }
        openai_tools.append(openai_tool)

    return openai_tools


def anthropic_to_langchain(request: AnthropicRequest) -> Dict[str, Any]:
    """
    Convert an AnthropicRequest to LangChain-compatible parameters.

    Args:
        request: AnthropicRequest object

    Returns:
        Dictionary with LangChain-compatible parameters:
        - messages: List of LangChain BaseMessage objects
        - tools: OpenAI-formatted tool definitions
        - temperature: Sampling temperature
        - max_tokens: Maximum tokens to generate

    Example:
        request = AnthropicRequest(
            model="oca/gpt-4.1",
            max_tokens=100,
            messages=[...]
        )

        → {
            "messages": [HumanMessage(...), AIMessage(...)],
            "tools": [...],
            "temperature": 0.7,
            "max_tokens": 100
        }
    """
    lc_messages = anthropic_to_langchain_messages(request.messages)
    openai_tools = anthropic_tools_to_openai_tools(request.tools)

    result = {
        "messages": lc_messages,
        "tools": openai_tools,
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
    }

    return result


def langchain_to_anthropic_response(
    lc_message: AIMessage,
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0
) -> AnthropicResponse:
    """
    Convert a LangChain AIMessage to an AnthropicResponse.

    Args:
        lc_message: LangChain AIMessage object
        model: Model identifier
        input_tokens: Input token count (optional)
        output_tokens: Output token count (optional)

    Returns:
        AnthropicResponse object

    Example:
        LangChain AIMessage with tool_calls:
        {
            "content": "Let me check the weather",
            "additional_kwargs": {
                "tool_calls": [{
                    "type": "function",
                    "id": "call_123",
                    "function": {"name": "weather", "arguments": '{"city": "Tokyo"}'}
                }]
            }
        }

        → Anthropic:
        {
            "content": [
                {"type": "text", "text": "Let me check the weather"},
                {"type": "tool_use", "id": "toolu_123", "name": "weather", "input": {"city": "Tokyo"}}
            ]
        }
    """
    content_blocks: List[AnthropicContentBlock] = []

    # Add text content if present
    if lc_message.content:
        content_blocks.append(
            AnthropicContentBlock(type="text", text=lc_message.content)
        )

    # Convert tool_calls from OpenAI format to Anthropic format
    tool_calls = lc_message.additional_kwargs.get("tool_calls")
    if tool_calls:
        for tool_call in tool_calls:
            if isinstance(tool_call, dict):
                function = tool_call.get("function", {})
                tool_use_id = tool_call.get("id", f"toolu_{uuid.uuid4().hex[:24]}")

                # Parse arguments JSON
                arguments_str = function.get("arguments", "{}")
                try:
                    arguments = json.loads(arguments_str)
                except json.JSONDecodeError:
                    arguments = {}

                content_blocks.append(
                    AnthropicContentBlock(
                        type="tool_use",
                        id=tool_use_id,
                        name=function.get("name", ""),
                        input=arguments
                    )
                )

    # Determine stop_reason
    stop_reason = "end_turn"
    if tool_calls:
        stop_reason = "tool_use"

    # Create usage
    usage = AnthropicUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens
    )

    # Create response
    response = AnthropicResponse(
        content=content_blocks,
        model=model,
        stop_reason=stop_reason,
        usage=usage
    )

    return response


def create_anthropic_error_response(
    error_type: str,
    message: str
) -> AnthropicErrorResponse:
    """
    Create an Anthropic-formatted error response.

    Args:
        error_type: Error type (e.g., "invalid_request_error", "authentication_error")
        message: Human-readable error message

    Returns:
        AnthropicErrorResponse object

    Example:
        create_anthropic_error_response(
            "invalid_request_error",
            "max_tokens is required"
        )

        → {
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "max_tokens is required"
            }
        }
    """
    return AnthropicErrorResponse(
        error={
            "type": error_type,
            "message": message
        }
    )
