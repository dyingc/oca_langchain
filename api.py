import os
import json
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from core.llm import OCAChatModel
from core.oauth2_token_manager import OCAOauth2TokenManager

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
    content: Union[str, List[Dict[str, str]]]

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False
    max_tokens: Optional[int] = None
    stream_options: Optional[Dict] = None

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
    print("--- Initializing core components ---")
    try:
        token_manager = OCAOauth2TokenManager(dotenv_path=".env", debug=True)
        chat_model = OCAChatModel.from_env(token_manager, debug=True)
        lifespan_objects["chat_model"] = chat_model
        print("--- Core components initialized successfully ---")
    except (FileNotFoundError, ValueError) as e:
        print(f"FATAL: Failed to initialize core components: {e}")
        # In this case, the application will not work properly.
        lifespan_objects["chat_model"] = None

    yield

    print("--- Cleaning up resources ---")
    lifespan_objects.clear()

# --- FastAPI App Initialization ---
app = FastAPI(lifespan=lifespan)

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

def convert_to_langchain_messages(messages: List[ChatMessage]) -> List[BaseMessage]:
    """
    Convert Pydantic models to LangChain Message objects.
    Handles cases where content is a string or a list of dictionaries.
    """
    lc_messages = []
    for msg in messages:
        content_str = ""
        if isinstance(msg.content, str):
            content_str = msg.content
        elif isinstance(msg.content, list):
            # Merge list of content chunks into a single string
            processed_parts = []
            for part in msg.content:
                if isinstance(part, dict) and "text" in part:
                    processed_parts.append(part["text"])
            content_str = "\n".join(processed_parts)

        if msg.role == "user":
            lc_messages.append(HumanMessage(content=content_str))
        elif msg.role == "assistant":
            lc_messages.append(AIMessage(content=content_str))
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
async def list_models():
    """
    Provides a LiteLLM-compatible endpoint for listing available models with detailed info.
    """
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

    lc_messages = convert_to_langchain_messages(request.messages)

    # --- Streaming response ---
    if request.stream:
        async def stream_generator():
            try:
                # Use astream for asynchronous streaming
                async for chunk in chat_model.astream(lc_messages, max_tokens=request.max_tokens):
                    if chunk.content:
                        stream_response = ChatCompletionStreamResponse(
                            model=request.model,
                            choices=[ChatCompletionStreamChoice(
                                index=0,
                                delta=DeltaMessage(content=chunk.content)
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
                max_tokens=request.max_tokens
            )

            completion_response = ChatCompletionResponse(
                model=request.model,
                choices=[ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response.content)
                )]
            )
            return completion_response

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # For direct execution and testing, use uvicorn:
    # Command line: uvicorn app:app --reload --port 8000
    print("To run this application, use the command:")
    print("uvicorn api:app --host 0.0.0.0 --port 8000")
