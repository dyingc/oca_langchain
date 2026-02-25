"""
OpenAI Response API Compatible Endpoints

This module provides FastAPI endpoints compatible with OpenAI's Response API,
allowing clients to use the Response API format with our OCA backend.

Endpoints:
- POST /v1/responses - Create a response (streaming and non-streaming)
- GET /v1/responses/{id} - Retrieve a stored response
- DELETE /v1/responses/{id} - Delete a stored response

Reference: https://platform.openai.com/docs/api-reference/responses
"""

import json
import asyncio
import os
from typing import Optional, Dict, Any
from fastapi import HTTPException, Header
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv

from langchain_core.messages import AIMessage

from models.responses_types import (
    ResponseRequest,
    Response,
    ResponseStatus,
    ResponseDeleted,
    OutputFunctionCall,
    OutputContentText,
    OutputMessage,
    generate_response_id,
    generate_item_id,
)
from converters.responses_converter import (
    response_request_to_langchain,
    langchain_to_response_output,
    create_response_error,
    create_response_created_event,
    create_output_item_added_event,
    create_output_text_delta_event,
    create_function_call_arguments_delta_event,
    create_output_item_done_event,
    create_response_completed_event,
    format_stream_event,
)
from core.logger import get_logger

logger = get_logger(__name__)


# --- Model Resolution ---
_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")


def _get_default_model() -> str:
    """Get LLM_MODEL_NAME from .env with runtime reload."""
    load_dotenv(_ENV_PATH, override=True)
    return os.getenv("LLM_MODEL_NAME", "").strip()


def resolve_model_name(incoming_model: str) -> str:
    """
    Resolve which model to use.

    If incoming model starts with 'oca/', use it directly.
    Otherwise, fall back to LLM_MODEL_NAME from .env.

    Args:
        incoming_model: The model name from the incoming request

    Returns:
        The model name to use
    """
    if incoming_model and incoming_model.strip().lower().startswith("oca/"):
        # Use incoming model if it already has oca/ prefix
        return incoming_model.strip()

    default_model = _get_default_model()

    # Fall back to default model from env
    if default_model:
        logger.info(f"[MODEL RESOLUTION] Incoming model '{incoming_model}' doesn't have oca/ prefix, using default: {default_model}")
        return default_model

    # No default set, use incoming as-is (will likely fail validation)
    return incoming_model


# --- In-Memory Response Storage ---
# For production, this should be replaced with a persistent store (Redis, DB, etc.)
_response_store: Dict[str, Response] = {}


def _get_chat_model():
    """Lazy import to avoid circular dependency."""
    from api import get_chat_model
    return get_chat_model()


def store_response(response: Response) -> None:
    """Store a response for later retrieval."""
    if response.id:
        _response_store[response.id] = response


def get_stored_response(response_id: str) -> Optional[Response]:
    """Retrieve a stored response by ID."""
    return _response_store.get(response_id)


def delete_stored_response(response_id: str) -> bool:
    """Delete a stored response. Returns True if deleted."""
    if response_id in _response_store:
        del _response_store[response_id]
        return True
    return False


# --- Validation ---

def validate_response_request(request: ResponseRequest) -> None:
    """
    Validate a Response API request.

    Args:
        request: ResponseRequest to validate

    Raises:
        HTTPException: If validation fails
    """
    if not request.model or request.model.strip() == "":
        raise HTTPException(
            status_code=400,
            detail={
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": "model: This field is required"
                }
            }
        )

    # Validate previous_response_id if provided
    if request.previous_response_id:
        prev_response = get_stored_response(request.previous_response_id)
        if not prev_response:
            raise HTTPException(
                status_code=404,
                detail={
                    "type": "error",
                    "error": {
                        "type": "not_found_error",
                        "message": f"previous_response_id: Response '{request.previous_response_id}' not found"
                    }
                }
            )


# --- Stream Generator ---

async def response_stream_generator(
    lc_messages: list,
    model: str,
    tools: list,
    max_tokens: Optional[int],
    response_id: str,
    previous_response_id: Optional[str] = None
):
    """
    Generate Response API compatible streaming events from LangChain stream.

    Args:
        lc_messages: LangChain messages
        model: Model identifier
        tools: Tool definitions
        max_tokens: Maximum tokens
        response_id: Unique response ID
        previous_response_id: Optional previous response ID

    Yields:
        Server-Sent Events (SSE) formatted strings
    """
    try:
        chat_model = _get_chat_model()
        chat_model.model = model

        # Debug log the incoming request details
        # Log messages in detail
        messages_debug = []
        for i, msg in enumerate(lc_messages):
            msg_dict = {
                "index": i,
                "type": type(msg).__name__,
                "role": getattr(msg, "role", "unknown") if hasattr(msg, "role") else None,
                "content_preview": str(getattr(msg, "content", ""))[:200] if getattr(msg, "content", "") else "",
                "has_tool_calls": bool(getattr(msg, "tool_calls", None) or getattr(msg, "additional_kwargs", {}).get("tool_calls")),
                "tool_call_id": getattr(msg, "tool_call_id", None) if hasattr(msg, "tool_call_id") else None,
            }
            messages_debug.append(msg_dict)
        logger.info(
            f"[RESPONSE API STREAM] model={model}, "
            f"messages={json.dumps(messages_debug, ensure_ascii=False)}, "
            f"tools_count={len(tools) if tools else 0}, "
            f"max_tokens={max_tokens}"
        )

        # Send response.created event
        sequence_number = 0  # Global sequence counter for events
        sequence_number += 1
        created_event = create_response_created_event(
            response_id=response_id,
            model=model,
            previous_response_id=previous_response_id,
            sequence_number=sequence_number
        )
        yield format_stream_event(created_event)

        # Track output state
        output_items = []
        current_text_block_index = 0
        current_output_index = 0
        full_text_content = ""
        tool_states: Dict[int, Dict[str, Any]] = {}

        # Start the first output message item
        message_item_id = generate_item_id("msg")
        output_items.append({
            "id": message_item_id,
            "type": "message",
            "role": "assistant",
            "status": "in_progress",
            "content": []
        })

        # Send output_item.added event for message
        sequence_number += 1
        added_event = create_output_item_added_event(
            output_index=0,
            item_type="message",
            item_id=message_item_id,
            sequence_number=sequence_number
        )
        yield format_stream_event(added_event)

        # Stream content
        async for chunk in chat_model.astream(lc_messages, max_tokens=max_tokens, tools=tools):
            # Handle different chunk formats
            if hasattr(chunk, "message") and chunk.message is not None:
                message = chunk.message
            else:
                message = chunk

            content_delta = getattr(message, "content", None)
            additional_kwargs = getattr(message, "additional_kwargs", {})
            tool_calls_delta = additional_kwargs.get("tool_calls")

            # Handle text content
            if content_delta:
                full_text_content += content_delta

                # Add text delta event with item_id and sequence_number
                sequence_number += 1
                delta_event = create_output_text_delta_event(
                    output_index=0,
                    content_index=0,
                    delta=content_delta,
                    item_id=message_item_id,
                    sequence_number=sequence_number
                )
                yield format_stream_event(delta_event)

            # Handle tool calls
            if tool_calls_delta:
                # Debug: log the raw tool_calls_delta
                logger.info(f"[DEBUG] tool_calls_delta: {json.dumps(tool_calls_delta, ensure_ascii=False)}")

                # Close the message item if we're transitioning to tool calls
                if output_items[0]["content"] or full_text_content:
                    # Add text content to message
                    if full_text_content:
                        output_items[0]["content"].append({
                            "type": "output_text",
                            "text": full_text_content
                        })
                    output_items[0]["status"] = "completed"

                    # Send output_item.done for message
                    sequence_number += 1
                    done_event = create_output_item_done_event(
                        output_index=0,
                        item=output_items[0],
                        sequence_number=sequence_number
                    )
                    yield format_stream_event(done_event)

                for tc in tool_calls_delta:
                    tc_index = tc.get("index", 0)
                    tc_id = tc.get("id")
                    tc_function = tc.get("function", {})
                    tc_name = tc_function.get("name", "")
                    tc_arguments = tc_function.get("arguments", "")

                    # Initialize tool state if new
                    if tc_index not in tool_states:
                        # Create new output item for function call
                        fc_item_id = generate_item_id("fc")
                        call_id = tc_id or f"call_{generate_item_id('call')}"

                        tool_states[tc_index] = {
                            "output_index": len(output_items),
                            "item_id": fc_item_id,
                            "call_id": call_id,
                            "name": tc_name,
                            "arguments": "",
                            "started": False
                        }

                        output_items.append({
                            "id": fc_item_id,
                            "type": "function_call",
                            "call_id": call_id,
                            "name": "",
                            "arguments": "",
                            "status": "in_progress"
                        })

                        # Send output_item.added event
                        sequence_number += 1
                        added_event = create_output_item_added_event(
                            output_index=len(output_items) - 1,
                            item_type="function_call",
                            item_id=fc_item_id,
                            sequence_number=sequence_number
                        )
                        yield format_stream_event(added_event)

                    state = tool_states[tc_index]

                    # Update name if provided
                    if tc_name and not state["name"]:
                        state["name"] = tc_name
                        output_items[state["output_index"]]["name"] = tc_name

                    # Stream arguments delta
                    if tc_arguments:
                        state["arguments"] += tc_arguments

                        sequence_number += 1
                        delta_event = create_function_call_arguments_delta_event(
                            output_index=state["output_index"],
                            call_id=state["call_id"],
                            delta=tc_arguments,
                            item_id=state["item_id"],
                            sequence_number=sequence_number
                        )
                        yield format_stream_event(delta_event)

        # Finalize output items

        # Finalize message item if it has content
        if full_text_content and not output_items[0]["content"]:
            output_items[0]["content"].append({
                "type": "output_text",
                "text": full_text_content
            })
        output_items[0]["status"] = "completed"

        # Send output_item.done for message
        sequence_number += 1
        done_event = create_output_item_done_event(
            output_index=0,
            item=output_items[0],
            sequence_number=sequence_number
        )
        yield format_stream_event(done_event)

        # Finalize all tool call items
        for tc_index, state in sorted(tool_states.items()):
            output_items[state["output_index"]]["arguments"] = state["arguments"]
            output_items[state["output_index"]]["status"] = "completed"

            sequence_number += 1
            done_event = create_output_item_done_event(
                output_index=state["output_index"],
                item=output_items[state["output_index"]],
                sequence_number=sequence_number
            )
            yield format_stream_event(done_event)

        # Calculate usage (rough estimate)
        input_tokens = sum(len(str(m.content).split()) for m in lc_messages) // 2
        output_tokens = len(full_text_content.split()) + sum(
            len(s["arguments"]) // 4 for s in tool_states.values()
        )

        usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens
        }

        # Send response.completed event
        # Clean output items for the final event
        clean_output = []
        for item in output_items:
            clean_item = {k: v for k, v in item.items() if v is not None}
            clean_output.append(clean_item)

        completed_event = create_response_completed_event(
            response_id=response_id,
            model=model,
            output=clean_output,
            usage=usage,
            previous_response_id=previous_response_id,
            sequence_number=sequence_number + 1
        )

        # Debug: Log the final output structure
        logger.info(f"[RESPONSE API] Completed event output: {json.dumps(clean_output, ensure_ascii=False)[:1000]}")

        yield format_stream_event(completed_event)

        # Store the response (for retrieval via GET /v1/responses/{id})
        response = Response(
            id=response_id,
            model=model,
            output=[],  # Will be populated from output_items
            status=ResponseStatus.COMPLETED,
            previous_response_id=previous_response_id
        )
        store_response(response)

    except Exception as e:
        logger.exception("Error in response_stream_generator")
        # Log more details about the error
        import traceback
        error_details = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "model": model,
            "tools_count": len(tools) if tools else 0,
        }
        # Check if it's an HTTP error with response body
        if hasattr(e, 'response'):
            try:
                error_details["response_status"] = getattr(e.response, 'status_code', None)
                error_details["response_body"] = getattr(e.response, 'text', None)[:500]
            except Exception:
                pass
        logger.error(f"[RESPONSE API STREAM ERROR] {json.dumps(error_details, ensure_ascii=False)}")

        error_event = {
            "type": "error",
            "error": {
                "type": "server_error",
                "message": str(e)
            }
        }
        yield format_stream_event(error_event)


# --- API Endpoints ---

async def create_response(
    request: ResponseRequest,
    authorization: Optional[str] = Header(None)
) -> Response:
    """
    POST /v1/responses endpoint.

    Create a model response. Supports both streaming and non-streaming responses.

    Args:
        request: ResponseRequest with model and input
        authorization: Optional Authorization header

    Returns:
        Response object (non-streaming) or StreamingResponse (streaming)
    """
    # Validate request
    validate_response_request(request)

    # Get chat model
    chat_model = _get_chat_model()

    # Resolve model name (use LLM_MODEL_NAME from env if set, otherwise use incoming)
    original_model = request.model
    request.model = resolve_model_name(request.model)

    # Check if model is available
    if request.model not in chat_model.available_models:
        raise HTTPException(
            status_code=404,
            detail={
                "type": "error",
                "error": {
                    "type": "not_found_error",
                    "message": f"Model '{request.model}' not found. Available models: {', '.join(chat_model.available_models)}"
                }
            }
        )

    # Log request
    model_log = request.model
    if original_model != request.model:
        model_log = f"{request.model} (from {original_model})"
    logger.info(
        f"[RESPONSE API] model={model_log}, "
        f"stream={request.stream}, "
        f"store={request.store}, "
        f"previous_response_id={request.previous_response_id}"
    )

    # Debug: Log raw request to see original function_call format
    try:
        with open("logs/debug_raw_request.json", "w") as f:
            # Get the raw request body
            raw = request.model_dump(mode='json')
            json.dump(raw, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"[DEBUG] Failed to log raw request: {e}")

    # Generate response ID
    response_id = generate_response_id()

    # Convert to LangChain format
    lc_params = response_request_to_langchain(request)
    lc_messages = lc_params["messages"]
    tools = lc_params["tools"]

    # Note: previous_response_id is kept for response metadata only.
    # We don't merge history - clients should manage their own conversation state
    # by sending complete message history in each request's input field.

    # Handle streaming vs non-streaming
    if request.stream:
        logger.info(f"[RESPONSE API] Starting streaming response {response_id}")
        return StreamingResponse(
            response_stream_generator(
                lc_messages=lc_messages,
                model=request.model,
                tools=tools,
                max_tokens=request.max_output_tokens,
                response_id=response_id,
                previous_response_id=request.previous_response_id
            ),
            media_type="text/event-stream"
        )

    else:
        # Non-streaming response
        try:
            logger.info(f"[RESPONSE API] Starting non-streaming invoke for {response_id}")

            # Use invoke for synchronous call
            response = await asyncio.to_thread(
                chat_model.invoke,
                lc_messages,
                max_tokens=request.max_output_tokens,
                tools=tools,
                tool_choice=lc_params.get("tool_choice")
            )

            # Convert to Response API format
            api_response = langchain_to_response_output(
                lc_message=response,
                model=request.model,
                response_id=response_id,
                previous_response_id=request.previous_response_id
            )

            # Store if requested (for retrieval via GET /v1/responses/{id})
            if request.store:
                store_response(api_response)

            logger.info(
                f"[RESPONSE API] Completed response {response_id}, "
                f"output_items={len(api_response.output)}"
            )

            return api_response

        except Exception as e:
            logger.exception(f"[RESPONSE API] Error during invoke: {e}")
            raise HTTPException(
                status_code=500,
                detail={
                    "type": "error",
                    "error": {
                        "type": "server_error",
                        "message": str(e)
                    }
                }
            )


async def get_response(response_id: str) -> Response:
    """
    GET /v1/responses/{id} endpoint.

    Retrieve a stored response by ID.

    Args:
        response_id: The ID of the response to retrieve

    Returns:
        Response object

    Raises:
        HTTPException: 404 if response not found
    """
    response = get_stored_response(response_id)
    if not response:
        raise HTTPException(
            status_code=404,
            detail={
                "type": "error",
                "error": {
                    "type": "not_found_error",
                    "message": f"Response '{response_id}' not found"
                }
            }
        )

    return response


async def delete_response(response_id: str) -> ResponseDeleted:
    """
    DELETE /v1/responses/{id} endpoint.

    Delete a stored response.

    Args:
        response_id: The ID of the response to delete

    Returns:
        ResponseDeleted confirmation

    Raises:
        HTTPException: 404 if response not found
    """
    if not delete_stored_response(response_id):
        raise HTTPException(
            status_code=404,
            detail={
                "type": "error",
                "error": {
                    "type": "not_found_error",
                    "message": f"Response '{response_id}' not found"
                }
            }
        )

    return ResponseDeleted(id=response_id, deleted=True)


# Note: These endpoints will be registered to the FastAPI app in api.py
