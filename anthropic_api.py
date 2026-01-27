"""
Anthropic Messages API Compatible Endpoints

This module provides FastAPI endpoints compatible with Anthropic's Messages API,
allowing clients to use the Anthropic SDK or format with our OCA backend.

Endpoints:
- POST /v1/messages - Create a message (non-streaming and streaming)
"""

import json
import time
import random
import asyncio
from typing import AsyncIterator
from fastapi import Request, HTTPException, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from langchain_core.messages import BaseMessage
from models.anthropic_types import (
    AnthropicRequest,
    AnthropicResponse,
    AnthropicUsage,
    AnthropicErrorResponse,
    AnthropicStreamMessageStart,
    AnthropicStreamContentBlockStart,
    AnthropicStreamContentBlockDelta,
    AnthropicStreamContentBlockStop,
    AnthropicStreamMessageDelta,
    AnthropicStreamMessageStop,
    AnthropicContentBlock,
)
from converters.anthropic_request_converter import (
    anthropic_to_langchain,
    langchain_to_anthropic_response,
    create_anthropic_error_response,
)
from core.logger import get_logger

logger = get_logger(__name__)


# Import get_chat_model here to avoid circular import
# It will be imported when needed (lazy import)
def _get_chat_model():
    """Lazy import to avoid circular dependency."""
    from api import get_chat_model
    return get_chat_model()


# --- Helper Functions ---

def validate_anthropic_request(request: AnthropicRequest) -> None:
    """
    Validate an Anthropic request according to Anthropic's API requirements.

    Required fields:
    - model: Must be a non-empty string
    - max_tokens: Must be > 0
    - messages: Must be non-empty

    Args:
        request: AnthropicRequest to validate

    Raises:
        HTTPException: If validation fails
    """
    if not request.model or request.model.strip() == "":
        raise HTTPException(
            status_code=400,
            detail=create_anthropic_error_response(
                "invalid_request_error",
                "model: This field is required"
            ).dict()
        )

    if request.max_tokens is None or request.max_tokens <= 0:
        raise HTTPException(
            status_code=400,
            detail=create_anthropic_error_response(
                "invalid_request_error",
                "max_tokens: This field is required and must be > 0"
            ).dict()
        )

    if not request.messages or len(request.messages) == 0:
        raise HTTPException(
            status_code=400,
            detail=create_anthropic_error_response(
                "invalid_request_error",
                "messages: Must provide at least one message"
            ).dict()
        )

    # Validate message roles
    valid_roles = {"user", "assistant", "system"}
    for msg in request.messages:
        if msg.role not in valid_roles:
            raise HTTPException(
                status_code=400,
                detail=create_anthropic_error_response(
                    "invalid_request_error",
                    f"messages: Invalid role '{msg.role}'. Must be one of: {valid_roles}"
                ).dict()
            )


# --- Stream Event Generators ---

def generate_message_start(message_id: str, model: str) -> str:
    """Generate message_start event"""
    event = AnthropicStreamMessageStart(
        message=AnthropicResponse(
            id=message_id,
            model=model,
            content=[],
            usage=AnthropicUsage(input_tokens=0, output_tokens=0)
        )
    )
    return f"event: message_start\ndata: {event.json(exclude_none=True)}\n\n"


def generate_content_block_start(index: int, content_block: dict) -> str:
    """Generate content_block_start event"""
    event = AnthropicStreamContentBlockStart(
        index=index,
        content_block=content_block
    )
    return f"event: content_block_start\ndata: {event.json(exclude_none=True)}\n\n"


def generate_content_block_delta(index: int, delta_type: str, text: str = None, partial_json: str = None) -> str:
    """Generate content_block_delta event"""
    delta = {"type": delta_type}
    if text is not None:
        delta["text"] = text
    if partial_json is not None:
        delta["partial_json"] = partial_json

    event = AnthropicStreamContentBlockDelta(
        index=index,
        delta=delta
    )
    return f"event: content_block_delta\ndata: {event.json(exclude_none=True)}\n\n"


def generate_content_block_stop(index: int) -> str:
    """Generate content_block_stop event"""
    event = AnthropicStreamContentBlockStop(index=index)
    return f"event: content_block_stop\ndata: {event.json(exclude_none=True)}\n\n"


def generate_message_delta(stop_reason: str, usage: AnthropicUsage) -> str:
    """Generate message_delta event"""
    event = AnthropicStreamMessageDelta(
        delta={"stop_reason": stop_reason, "stop_sequence": None},
        usage=usage
    )
    return f"event: message_delta\ndata: {event.json(exclude_none=True)}\n\n"


def generate_message_stop() -> str:
    """Generate message_stop event"""
    event = AnthropicStreamMessageStop()
    return f"event: message_stop\ndata: {event.json(exclude_none=True)}\n\n"


async def anthropic_stream_generator(
    lc_messages: list[BaseMessage],
    model: str,
    tools: list,
    max_tokens: int,
    message_id: str
) -> AsyncIterator[str]:
    """
    Generate Anthropic-compatible streaming events from LangChain stream.

    Handles both text content and tool_calls, converting OpenAI format tool_calls
    to Anthropic format tool_use content blocks.

    Args:
        lc_messages: LangChain messages
        model: Model identifier
        tools: Tool definitions (OpenAI format)
        max_tokens: Maximum tokens
        message_id: Unique message ID

    Yields:
        Server-Sent Events (SSE) formatted strings
    """
    try:
        chat_model = _get_chat_model()
        chat_model.model = model

        # Send message_start event
        yield generate_message_start(message_id, model)

        # Start streaming from LangChain
        content_buffer = ""
        block_index = 0
        text_block_started = False
        text_block_has_content = False

        # Track tool_calls state
        # Key: tool index (from OpenAI format), Value: dict with id, name, arguments_buffer, block_index, started
        tool_states: dict[int, dict] = {}
        stop_reason = "end_turn"

        # Stream content
        async for chunk in chat_model.astream(lc_messages, max_tokens=max_tokens, tools=tools):
            # The chunk can be either:
            # - ChatGenerationChunk (with .message attribute containing AIMessageChunk)
            # - AIMessageChunk directly
            # We need to handle both cases
            if hasattr(chunk, "message") and chunk.message is not None:
                message = chunk.message
            else:
                # chunk itself is the message (AIMessageChunk)
                message = chunk

            # Get content and additional_kwargs from the message
            content_delta = getattr(message, "content", None)
            additional_kwargs = getattr(message, "additional_kwargs", {})
            if content_delta:
                # Start text block on first text content
                if not text_block_started:
                    yield generate_content_block_start(
                        index=block_index,
                        content_block={"type": "text", "text": ""}
                    )
                    text_block_started = True

                content_buffer += content_delta
                text_block_has_content = True
                yield generate_content_block_delta(
                    index=block_index,
                    delta_type="text_delta",
                    text=content_delta
                )

            # Handle tool_calls from additional_kwargs (already extracted above)
            tool_calls_delta = additional_kwargs.get("tool_calls")

            if tool_calls_delta:
                # Close text block if it was started and has content
                if text_block_started and text_block_has_content:
                    yield generate_content_block_stop(block_index)
                    block_index += 1
                    text_block_started = False
                    text_block_has_content = False

                for tc in tool_calls_delta:
                    tc_index = tc.get("index", 0)
                    tc_id = tc.get("id")
                    tc_type = tc.get("type")
                    tc_function = tc.get("function", {})
                    tc_name = tc_function.get("name")
                    tc_arguments = tc_function.get("arguments", "")

                    # Initialize tool state if this is a new tool
                    if tc_index not in tool_states:
                        tool_states[tc_index] = {
                            "id": tc_id,
                            "name": tc_name,
                            "arguments_buffer": "",
                            "block_index": block_index + tc_index,
                            "started": False
                        }

                    state = tool_states[tc_index]

                    # Update id and name if provided (they come in the first chunk)
                    if tc_id and not state["id"]:
                        state["id"] = tc_id
                    if tc_name and not state["name"]:
                        state["name"] = tc_name

                    # Send content_block_start for this tool if not started
                    if not state["started"] and state["id"] and state["name"]:
                        # Convert OpenAI tool call ID to Anthropic format
                        anthropic_id = state["id"]
                        if anthropic_id.startswith("call_"):
                            anthropic_id = "toolu_" + anthropic_id[5:]

                        yield generate_content_block_start(
                            index=state["block_index"],
                            content_block={
                                "type": "tool_use",
                                "id": anthropic_id,
                                "name": state["name"],
                                "input": {}
                            }
                        )
                        state["started"] = True

                    # Stream argument fragments as input_json_delta
                    if tc_arguments and state["started"]:
                        state["arguments_buffer"] += tc_arguments
                        yield generate_content_block_delta(
                            index=state["block_index"],
                            delta_type="input_json_delta",
                            partial_json=tc_arguments
                        )

                # Set stop_reason to tool_use if we have tool calls
                stop_reason = "tool_use"

        # Close text block if it was started but not closed
        if text_block_started:
            yield generate_content_block_stop(block_index)
            block_index += 1

        # Close all tool blocks
        for tc_index, state in sorted(tool_states.items()):
            if state["started"]:
                yield generate_content_block_stop(state["block_index"])

        # Send message_delta with usage
        # Note: Accurate token counting requires backend support
        usage = AnthropicUsage(
            input_tokens=0,  # Backend should provide this
            output_tokens=len(content_buffer.split()) + sum(
                len(s["arguments_buffer"]) for s in tool_states.values()
            ) // 4  # Rough estimate
        )
        yield generate_message_delta(stop_reason, usage)

        # Send message_stop
        yield generate_message_stop()

    except Exception as e:
        logger.exception("Error in anthropic_stream_generator")
        # Send error event
        error_response = create_anthropic_error_response("api_error", str(e))
        yield f"event: error\ndata: {error_response.json()}\n\n"


# --- API Endpoints ---

async def create_message(
    request: AnthropicRequest,
    x_api_key: str = Header(None),
    anthropic_version: str = Header(None)
):
    """
    Anthropic-compatible /v1/messages endpoint.

    Supports both non-streaming and streaming responses.

    Headers:
        x-api-key: API key (optional, for validation)
        anthropic-version: API version (recommended: 2023-06-01)

    Args:
        request: AnthropicRequest
        x_api_key: API key from header
        anthropic_version: API version from header

    Returns:
        AnthropicResponse (non-streaming) or StreamingResponse (streaming)
    """
    # Log version warning if missing
    if anthropic_version is None:
        logger.warning("[ANTHROPIC] Missing anthropic-version header")

    # Validate request
    validate_anthropic_request(request)

    # Get chat model
    chat_model = _get_chat_model()

    # Check if model is available
    if request.model not in chat_model.available_models:
        raise HTTPException(
            status_code=404,
            detail=create_anthropic_error_response(
                "not_found_error",
                f"Model '{request.model}' not found. Available models: {', '.join(chat_model.available_models)}"
            ).dict()
        )

    # Log request
    logger.info(
        f"[ANTHROPIC REQUEST] model={request.model}, "
        f"max_tokens={request.max_tokens}, "
        f"stream={request.stream}, "
        f"messages={len(request.messages)}, "
        f"tools={len(request.tools) if request.tools else 0}"
    )

    # Convert to LangChain format
    lc_params = anthropic_to_langchain(request)
    lc_messages = lc_params["messages"]
    tools = lc_params["tools"]

    # Generate message ID
    message_id = f"msg_{''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=24))}"

    # Handle streaming vs non-streaming
    if request.stream:
        logger.info(f"[ANTHROPIC] Starting streaming response for message {message_id}")
        return StreamingResponse(
            anthropic_stream_generator(
                lc_messages=lc_messages,
                model=request.model,
                tools=tools,
                max_tokens=request.max_tokens,
                message_id=message_id
            ),
            media_type="text/event-stream"
        )

    else:
        # Non-streaming response
        try:
            logger.info(f"[ANTHROPIC] Starting non-streaming invoke for message {message_id}")

            # Use invoke for synchronous call
            response = await asyncio.to_thread(
                chat_model.invoke,
                lc_messages,
                max_tokens=request.max_tokens,
                tools=tools
            )

            # Convert to Anthropic format
            anthropic_resp = langchain_to_anthropic_response(
                lc_message=response,
                model=request.model,
                input_tokens=0,  # Backend should provide
                output_tokens=0  # Backend should provide
            )
            anthropic_resp.id = message_id

            logger.info(
                f"[ANTHROPIC RESPONSE] message_id={message_id}, "
                f"stop_reason={anthropic_resp.stop_reason}, "
                f"content_blocks={len(anthropic_resp.content)}"
            )

            return anthropic_resp

        except Exception as e:
            logger.exception(f"[ANTHROPIC] Error during invoke: {e}")
            raise HTTPException(
                status_code=500,
                detail=create_anthropic_error_response("api_error", str(e)).dict()
            )


# Note: This endpoint will be registered to the FastAPI app in api.py
# or in a separate initialization file
