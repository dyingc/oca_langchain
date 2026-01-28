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


def validate_and_fix_anthropic_message_sequence(
    anthropic_messages: List[AnthropicMessage]
) -> List[AnthropicMessage]:
    """
    Validate and fix incomplete tool_use sequences in Anthropic message history.

    When an assistant message contains tool_use blocks, it must be immediately
    followed by user messages containing tool_result blocks for each tool_use_id.
    No other assistant messages should appear in between.

    This function detects violations and removes incomplete tool_use blocks
    and their orphaned tool_results.

    Args:
        anthropic_messages: List of AnthropicMessage objects to validate

    Returns:
        Cleaned list of AnthropicMessage objects with valid tool_use sequences
    """
    cleaned_messages = []
    i = 0

    while i < len(anthropic_messages):
        msg = anthropic_messages[i]

        # Check if this is an assistant message with tool_use blocks
        if msg.role == "assistant" and isinstance(msg.content, list):
            tool_use_blocks = [b for b in msg.content if b.type == "tool_use"]
            num_tool_uses = len(tool_use_blocks)

            if num_tool_uses == 0:
                # No tool_use blocks, keep message as-is
                cleaned_messages.append(msg)
                i += 1
                continue

            # Look ahead to check if we have all required tool_results
            j = i + 1
            found_tool_results = 0
            valid_sequence = True

            while j < len(anthropic_messages) and found_tool_results < num_tool_uses:
                next_msg = anthropic_messages[j]

                # Check if next message is a user message with tool_result blocks
                if next_msg.role == "user" and isinstance(next_msg.content, list):
                    tool_result_blocks = [b for b in next_msg.content if b.type == "tool_result"]
                    if tool_result_blocks:
                        # Count the number of tool_result blocks, not messages
                        found_tool_results += len(tool_result_blocks)
                        j += 1
                    else:
                        # User message without tool_result - might be a new user message
                        valid_sequence = False
                        break
                elif next_msg.role == "user":
                    # User message with text content - break the sequence
                    valid_sequence = False
                    break
                else:
                    # Non-user message - break the sequence
                    valid_sequence = False
                    break

            # If we found all tool_results consecutively, keep the sequence
            if valid_sequence and found_tool_results == num_tool_uses:
                cleaned_messages.append(msg)
                # Add all the tool_result messages
                for k in range(i + 1, j):
                    cleaned_messages.append(anthropic_messages[k])
                i = j
            else:
                # Invalid sequence: remove tool_use blocks from assistant message
                # and keep only text content
                from models.anthropic_types import AnthropicContentBlock

                text_blocks = [b for b in msg.content if b.type == "text"]
                if text_blocks:
                    # Convert to text-only message
                    text_content = "\n".join([b.text or "" for b in text_blocks])
                    cleaned_messages.append(
                        AnthropicMessage(role=msg.role, content=text_content)
                    )
                # Skip orphaned tool_result messages
                i += 1
                while i < len(anthropic_messages):
                    next_msg = anthropic_messages[i]
                    if next_msg.role == "user" and isinstance(next_msg.content, list):
                        tool_result_blocks = [b for b in next_msg.content if b.type == "tool_result"]
                        if tool_result_blocks:
                            # This is an orphaned tool_result, skip it
                            i += 1
                            continue
                    break
        else:
            # Not an assistant message with tool_use, keep as-is
            cleaned_messages.append(msg)
            i += 1

    return cleaned_messages


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
                lc_messages.append(HumanMessage(content=msg.content))
            elif isinstance(msg.content, list):
                # Separate tool_result blocks from other content blocks
                tool_result_blocks = [b for b in msg.content if b.type == "tool_result"]
                other_blocks = [b for b in msg.content if b.type != "tool_result"]

                # Process other content blocks (text, image) as HumanMessage
                if other_blocks:
                    text_parts = []
                    for block in other_blocks:
                        if block.type == "text":
                            text_parts.append(block.text or "")
                        elif block.type == "image":
                            # Future: handle image blocks
                            text_parts.append("[Image]")
                    if text_parts:
                        lc_messages.append(HumanMessage(content="\n".join(text_parts)))

                # Process tool_result blocks as ToolMessage (OpenAI format)
                for block in tool_result_blocks:
                    tool_use_id = block.tool_use_id or ""
                    tool_content = block.content

                    # Extract content string from tool_result
                    if isinstance(tool_content, str):
                        content_str = tool_content
                    elif isinstance(tool_content, list):
                        # Extract text from tool result content blocks
                        content_parts = []
                        for sub_block in tool_content:
                            if isinstance(sub_block, dict) and sub_block.get("type") == "text":
                                content_parts.append(sub_block.get("text", ""))
                            elif isinstance(sub_block, str):
                                content_parts.append(sub_block)
                        content_str = "\n".join(content_parts)
                    else:
                        content_str = str(tool_content) if tool_content else ""

                    # Create ToolMessage for OpenAI compatibility
                    lc_messages.append(ToolMessage(
                        content=content_str,
                        tool_call_id=tool_use_id
                    ))
            else:
                lc_messages.append(HumanMessage(content=str(msg.content)))

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
    # Validate and fix message sequence before conversion
    validated_messages = validate_and_fix_anthropic_message_sequence(request.messages)
    lc_messages = anthropic_to_langchain_messages(validated_messages)
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
