import os
import json
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage, ToolMessage
from core.llm import OCAChatModel
from core.oauth2_token_manager import OCAOauth2TokenManager
from core.logger import get_logger

# Import Anthropic API endpoints
from anthropic_api import create_message

logger = get_logger(__name__)

# --- Pydantic Models for OpenAI Compatibility ---

class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(__import__('time').time()))
    owned_by: str = "owner"

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard]

class ChatMessage(BaseModel):
    role: str
    content: Optional[Union[str, List[Dict[str, str]]]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None  # Present on assistant deltas
    tool_call_id: Optional[str] = None                 # Present on tool role messages

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False
    max_tokens: Optional[int] = None
    stream_options: Optional[Dict] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    tools: Optional[List[Dict[str, Any]]] = None

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = "stop"

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{''.join(__import__('random').choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=29))}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(__import__('time').time()))
    model: str
    choices: List[ChatCompletionChoice]
    usage: Optional[Dict[str, int]] = None

class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

class ChatCompletionStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[str] = None

class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{''.join(__import__('random').choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=29))}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(__import__('time').time()))
    model: str
    choices: List[ChatCompletionStreamChoice]

# --- Global Objects ---
# Use a dictionary to store objects during the application lifecycle
lifespan_objects = {}

# --- FastAPI Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize core components at application startup and clean up on shutdown.
    """
    logger.info("--- Initializing core components ---")
    try:
        token_manager = OCAOauth2TokenManager(dotenv_path=".env", debug=True)
        chat_model = OCAChatModel.from_env(token_manager, debug=True)
        lifespan_objects["chat_model"] = chat_model
        logger.info("--- Core components initialized successfully ---")
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.exception("FATAL: Failed to initialize core components")
        # In this case, the application will not work properly.
        lifespan_objects["chat_model"] = None

    yield

    logger.info("--- Cleaning up resources ---")
    lifespan_objects.clear()

# --- FastAPI App Initialization ---
app = FastAPI(lifespan=lifespan)

# --- Validation Error Handler ---
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Log validation errors with the original request body for debugging.
    """
    body = await request.body()
    try:
        body_json = json.loads(body)
    except:
        body_json = body.decode("utf-8", errors="replace")

    # Save to file for debugging
    with open("logs/validation_error_request.json", "w") as f:
        json.dump({
            "url": str(request.url),
            "method": request.method,
            "body": body_json,
            "errors": exc.errors()
        }, f, ensure_ascii=False, indent=2)

    logger.error(f"[VALIDATION ERROR] {exc.errors()}")
    # Re-raise to return normal 422 response
    raise exc

# --- Helper Functions ---
def get_chat_model() -> OCAChatModel:
    """
    Get the initialized chat_model instance. Raises exception if unavailable.
    """
    model = lifespan_objects.get("chat_model")
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Chat model is not available due to initialization failure. Check server logs."
        )
    return model

# NOTE: This function has been deprecated in favor of unified validation in core/llm.py
# The weight-based algorithm in _validate_tool_call_sequences() handles all cases more robustly
# def validate_and_fix_message_sequence(messages: List[ChatMessage]) -> List[ChatMessage]:
#     """
#     Validate and fix incomplete tool_call sequences in the message history.
#
#     OpenAI API requires that when an assistant message contains tool_calls,
#     it must be immediately followed by tool response messages for each tool_call_id.
#     No other messages (user/assistant/system) should appear in between.
#
#     This function detects violations and removes the incomplete tool_calls
#     and their orphaned tool responses.
#
#     Args:
#         messages: List of ChatMessage objects to validate
#
#     Returns:
#         Cleaned list of ChatMessage objects with valid tool_call sequences
#     """
#     cleaned_messages = []
#     i = 0
#     skip_tool_messages = False  # Flag to skip orphaned tool messages
#
#     while i < len(messages):
#         msg = messages[i]
#
#         # Skip orphaned tool messages if flag is set
#         if skip_tool_messages and msg.role == "tool":
#             logger.warning(
#                 f"[MESSAGE VALIDATION] Skipping orphaned tool response at message {i}"
#             )
#             i += 1
#             continue
#
#         # Check if this is an assistant message with tool_calls
#         if msg.role == "assistant" and msg.tool_calls:
#             # Extract all tool_call_ids
#             tool_call_ids = [tc.get("id") for tc in msg.tool_calls if tc.get("id")]
#             num_tool_calls = len(tool_call_ids)
#
#             if num_tool_calls == 0:
#                 # No valid tool_call_ids, just add the message without tool_calls
#                 cleaned_messages.append(ChatMessage(role=msg.role, content=msg.content, tool_calls=None))
#                 i += 1
#                 continue
#
#             # Look ahead to check if we have all required tool responses
#             j = i + 1
#             found_tool_responses = 0
#             valid_sequence = True
#
#             while j < len(messages) and found_tool_responses < num_tool_calls:
#                 next_msg = messages[j]
#
#                 if next_msg.role == "tool" and next_msg.tool_call_id in tool_call_ids:
#                     found_tool_responses += 1
#                     j += 1
#                 elif next_msg.role == "tool":
#                     # Tool response for a different tool_call_id - still count it
#                     found_tool_responses += 1
#                     j += 1
#                 else:
#                     # Found a non-tool message before collecting all tool responses
#                     valid_sequence = False
#                     break
#
#             # If we found all tool responses consecutively, keep the sequence
#             if valid_sequence and found_tool_responses == num_tool_calls:
#                 cleaned_messages.append(msg)
#                 # Add all the tool response messages
#                 for k in range(i + 1, j):
#                     cleaned_messages.append(messages[k])
#                 i = j
#                 skip_tool_messages = False  # Reset flag
#             else:
#                 # Invalid sequence: remove tool_calls from assistant message
#                 # and mark for skipping orphaned tool responses
#                 logger.warning(
#                     f"[MESSAGE VALIDATION] Incomplete tool_calls detected at message {i}. "
#                     f"Expected {num_tool_calls} tool responses, but found non-tool message interrupting. "
#                     f"Removing tool_calls from assistant message."
#                 )
#                 cleaned_messages.append(ChatMessage(role=msg.role, content=msg.content, tool_calls=None))
#                 i += 1
#                 skip_tool_messages = True  # Set flag to skip subsequent tool messages
#                 continue
#         else:
#             # Not an assistant message with tool_calls, keep as-is
#             cleaned_messages.append(msg)
#             i += 1
#             # Note: Don't reset skip_tool_messages here - it will be reset naturally
#             # when we skip all orphaned tool messages and encounter the next assistant/user message
#
#     return cleaned_messages

def convert_to_langchain_messages(messages: List[ChatMessage]) -> List[BaseMessage]:
    """
    Convert API ChatMessage list to LangChain BaseMessage list, preserving
    tool_calls in assistant messages and translating tool role messages.
    """
    lc_messages: List[BaseMessage] = []
    for msg in messages:
        # Normalize content
        if msg.content is None:
            content_str = ""
        elif isinstance(msg.content, str):
            content_str = msg.content
        elif isinstance(msg.content, list):
            parts = [part.get("text", "") for part in msg.content if isinstance(part, dict)]
            content_str = "\n".join(parts)
        else:
            content_str = str(msg.content)

        # Role mapping
        if msg.role == "user":
            lc_messages.append(HumanMessage(content=content_str))

        elif msg.role == "assistant":
            # 仅当存在 tool_calls 时才传 additional_kwargs，避免空 dict 触发验证异常
            if msg.tool_calls:
                lc_messages.append(
                    AIMessage(content=content_str, additional_kwargs={"tool_calls": msg.tool_calls})
                )
            else:
                lc_messages.append(AIMessage(content=content_str))

        elif msg.role == "tool":
            # 当 tool_call_id 为空时省略该字段
            if msg.tool_call_id:
                lc_messages.append(ToolMessage(content=content_str, tool_call_id=msg.tool_call_id))
            else:
                lc_messages.append(ToolMessage(content=content_str))

        elif msg.role == "system":
            lc_messages.append(SystemMessage(content=content_str))

    return lc_messages

# --- API Endpoints ---
@app.get("/v1/models", response_model=ModelList)
async def list_models():
    """
    Provides an OpenAI-compatible endpoint for listing available models.
    """
    chat_model = get_chat_model()
    model_cards = [ModelCard(id=model_id) for model_id in chat_model.available_models]
    return ModelList(data=model_cards)

class LiteLLMParams(BaseModel):
    model: str
    # Add other litellm_params fields as needed

class ModelInfo(BaseModel):
    id: str
    db_model: bool = False
    key: Optional[str] = None
    max_tokens: Optional[int] = None
    max_input_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None
    input_cost_per_token: Optional[float] = None
    input_cost_per_character: Optional[float] = None
    output_cost_per_token: Optional[float] = None
    output_cost_per_character: Optional[float] = None
    litellm_provider: Optional[str] = None
    mode: Optional[str] = None
    # Add any custom fields you want to include

class ModelData(BaseModel):
    model_name: str
    litellm_params: LiteLLMParams
    model_info: ModelInfo

class ModelInfoList(BaseModel):
    data: List[ModelData]

@app.get("/v1/model/info", response_model=ModelInfoList)
async def list_models_info():
    """
    Provides a LiteLLM-compatible endpoint for listing available models with detailed info.
    """
    try:
        chat_model = get_chat_model()

        model_data_list = []
        for model_id in chat_model.available_models:
            model_data = ModelData(
                model_name=model_id,
                litellm_params=LiteLLMParams(
                    model=model_id
                ),
                model_info=ModelInfo(
                    id=model_id,
                    db_model=False,
                    key=model_id,
                    mode="chat",  # or "completion" depending on your models
                    litellm_provider="oca",  # your provider name
                    # Add token limits and pricing if available:
                    # max_tokens=4096,
                    # max_input_tokens=8192,
                    # max_output_tokens=4096,
                    # input_cost_per_token=0.00003,
                    # output_cost_per_token=0.00006,
                )
            )
            model_data_list.append(model_data)

        return ModelInfoList(data=model_data_list)
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.exception("Error in list_models_info")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/spend/calculate")
async def spend_calculate(request: Request):
    """
    Dummy endpoint for compatibility with OpenAI API.
    This endpoint does not perform any actual calculations.
    It simply returns a placeholder response.
    """
    # Parse the request body
    body = await request.json()

    return {
        "id": "spend-calc-12345",
        "object": "spend.calculation",
        "model": body.get("model", "unknown"),
        "usage": {
            "prompt_tokens": body.get("prompt_tokens", 0),
            "completion_tokens": body.get("completion_tokens", 0),
            "total_tokens": body.get("total_tokens", 0)
        },
        "result": {
            "cost": 0.0,
            "currency": "USD"
        }
    }

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """
    Provides an OpenAI-compatible chat completion endpoint, supporting both streaming and non-streaming responses.
    """
    chat_model = get_chat_model()

    # Check if the requested model is available
    if request.model not in chat_model.available_models:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{request.model}' not found. Available models: {', '.join(chat_model.available_models)}"
        )

    # Update the model instance with parameters from the request
    chat_model.model = request.model
    chat_model.temperature = request.temperature

    # NOTE: Validation moved to unified _validate_tool_call_sequences() in core/llm.py
    # No need to pre-validate here - it will be done automatically in _stream()/_astream()
    lc_messages = convert_to_langchain_messages(request.messages)

    # --- Streaming response ---
    if request.stream:
        async def stream_generator():
            try:
                # Use astream for asynchronous streaming
                async for chunk in chat_model.astream(lc_messages, max_tokens=request.max_tokens, tool_choice=request.tool_choice, tools=request.tools):
                    # Support both content tokens and tool_calls deltas
                    content_delta = getattr(chunk, "content", None)
                    tool_calls_delta = None
                    try:
                        tool_calls_delta = getattr(getattr(chunk, "message", None), "additional_kwargs", {}).get("tool_calls")
                    except Exception:
                        tool_calls_delta = None

                    if content_delta or tool_calls_delta:
                        stream_response = ChatCompletionStreamResponse(
                            model=request.model,
                            choices=[ChatCompletionStreamChoice(
                                index=0,
                                delta=DeltaMessage(content=content_delta, tool_calls=tool_calls_delta)
                            )]
                        )
                        yield f"data: {stream_response.json()}\n\n"

                # Send final [DONE] signal
                final_chunk = ChatCompletionStreamResponse(
                    model=request.model,
                    choices=[ChatCompletionStreamChoice(
                        index=0,
                        delta=DeltaMessage(),
                        finish_reason="stop"
                    )]
                )
                yield f"data: {final_chunk.json()}\n\n"
                yield "data: [DONE]\n\n"

            except Exception as e:
                print(f"Error during streaming: {e}")
                # Optionally send an error info chunk
                error_response = {
                    "error": {"message": "An error occurred during streaming.", "type": "server_error"}
                }
                yield f"data: {json.dumps(error_response)}\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    # --- Non-streaming response ---
    else:
        try:
            # Use invoke for synchronous call (run within asyncio.to_thread)
            response = await asyncio.to_thread(
                chat_model.invoke,
                lc_messages,
                max_tokens=request.max_tokens,
                tool_choice=request.tool_choice,
                tools=request.tools
            )

            # Extract potential tool_calls from model response
            tool_calls = None
            try:
                tool_calls = response.additional_kwargs.get("tool_calls")  # type: ignore[attr-defined]
            except Exception:
                tool_calls = None

            completion_response = ChatCompletionResponse(
                model=request.model,
                choices=[ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response.content, tool_calls=tool_calls)
                )]
            )
            return completion_response

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

# --- Anthropic-Compatible Endpoints ---

from models.anthropic_types import AnthropicRequest

@app.post("/v1/messages")
async def anthropic_create_message(
    request: AnthropicRequest
):
    """
    Anthropic-compatible /v1/messages endpoint.

    Provides compatibility with Anthropic's Messages API, supporting both
    non-streaming and streaming responses. This endpoint allows clients
    using the Anthropic SDK to communicate with our OCA backend.

    The endpoint handles:
    - Message format conversion (Anthropic ↔ LangChain)
    - Tool calls (Anthropic format ↔ OpenAI format)
    - Streaming responses (Anthropic SSE format)

    Args:
        request: AnthropicRequest with Anthropic-formatted messages

    Returns:
        AnthropicResponse (non-streaming) or StreamingResponse (streaming)
    """
    return await create_message(request)


# --- OpenAI Response API Endpoints ---

from responses_api import create_response, get_response, delete_response
from responses_passthrough import (
    is_passthrough_enabled,
    create_response_passthrough
)
from models.responses_types import ResponseRequest


@app.post("/v1/responses")
async def responses_create(
    request: Request
):
    """
    OpenAI Response API /v1/responses endpoint.

    If LLM_RESPONSES_API_URL is configured, requests are passed through directly
    to the backend LLM that natively supports the Response API format.

    Otherwise, provides compatibility by converting between Response API format
    and LangChain messages.

    Supports both non-streaming and streaming responses.

    Args:
        request: FastAPI Request object (raw request for passthrough flexibility)

    Returns:
        Response (non-streaming) or StreamingResponse (streaming)
    """
    # Get the raw request body
    body_bytes = await request.body()
    try:
        request_body = json.loads(body_bytes)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail={
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": "Invalid JSON in request body"
                }
            }
        )

    # Check if passthrough mode is enabled
    if is_passthrough_enabled():
        logger.info("[RESPONSES API] Using passthrough mode (LLM_RESPONSES_API_URL configured)")
        return await create_response_passthrough(request_body)

    # Fall back to the conversion-based approach
    # Parse the request into our Pydantic model
    try:
        response_request = ResponseRequest(**request_body)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": str(e)
                }
            }
        )

    return await create_response(response_request)


@app.get("/v1/responses/{response_id}")
async def responses_get(response_id: str):
    """
    Retrieve a stored response by ID.

    Args:
        response_id: The ID of the response to retrieve

    Returns:
        Response object
    """
    return await get_response(response_id)


@app.delete("/v1/responses/{response_id}")
async def responses_delete(response_id: str):
    """
    Delete a stored response.

    Args:
        response_id: The ID of the response to delete

    Returns:
        ResponseDeleted confirmation
    """
    return await delete_response(response_id)


if __name__ == "__main__":
    # For direct execution and testing, use uvicorn:
    # Command line: uvicorn app:app --reload --port 8000
    print("To run this application, use the command:")
    print("uvicorn api:app --host 0.0.0.0 --port 8000")
