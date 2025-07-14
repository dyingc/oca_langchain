import os
import json
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from oca.llm import OCAChatModel
from oca.oauth2_token_manager import OCAOauth2TokenManager

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
# 使用一个字典来存储生命周期中的对象
lifespan_objects = {}

# --- FastAPI Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    在应用启动时初始化核心组件，在关闭时清理。
    """
    print("--- Initializing core components ---")
    try:
        token_manager = OCAOauth2TokenManager(dotenv_path=".env", debug=True)
        chat_model = OCAChatModel.from_env(token_manager, debug=True)
        lifespan_objects["chat_model"] = chat_model
        print("--- Core components initialized successfully ---")
    except (FileNotFoundError, ValueError) as e:
        print(f"FATAL: Failed to initialize core components: {e}")
        # 在这种情况下，应用将无法正常工作
        lifespan_objects["chat_model"] = None

    yield

    print("--- Cleaning up resources ---")
    lifespan_objects.clear()

# --- FastAPI App Initialization ---
app = FastAPI(lifespan=lifespan)

# --- Helper Functions ---
def get_chat_model() -> OCAChatModel:
    """
    获取已初始化的 chat_model 实例，如果失败则抛出异常。
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
    将 Pydantic 模型转换为 LangChain 的 Message 对象。
    能够处理 content 是字符串或字典列表的情况。
    """
    lc_messages = []
    for msg in messages:
        content_str = ""
        if isinstance(msg.content, str):
            content_str = msg.content
        elif isinstance(msg.content, list):
            # 将内容块列表合并为一个字符串
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
    提供兼容 OpenAI 的模型列表端点。
    """
    chat_model = get_chat_model()
    model_cards = [ModelCard(id=model_id) for model_id in chat_model.available_models]
    return ModelList(data=model_cards)

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """
    提供兼容 OpenAI 的聊天补全端点，支持流式和非流式响应。
    """
    chat_model = get_chat_model()

    # 检查请求的模型是否可用
    if request.model not in chat_model.available_models:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{request.model}' not found. Available models: {', '.join(chat_model.available_models)}"
        )

    # 更新模型实例以使用请求的参数
    chat_model.model = request.model
    chat_model.temperature = request.temperature

    lc_messages = convert_to_langchain_messages(request.messages)

    # --- 流式响应 ---
    if request.stream:
        async def stream_generator():
            try:
                # 使用 astream 进行异步流式处理
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

                # 发送最后的 [DONE] 信号
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
                # 可以在这里发送一个错误信息的 chunk
                error_response = {
                    "error": {"message": "An error occurred during streaming.", "type": "server_error"}
                }
                yield f"data: {json.dumps(error_response)}\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    # --- 非流式响应 ---
    else:
        try:
            # 使用 invoke 进行同步调用（在异步函数中通过 asyncio.to_thread 运行）
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
    # 为了直接运行进行测试，可以使用 uvicorn
    # 命令行: uvicorn app:app --reload --port 8000
    print("To run this application, use the command:")
    print("uvicorn api:app --host 0.0.0.0 --port 8000")
