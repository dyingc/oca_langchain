import os
import json
import requests
import httpx
from typing import Any, List, Mapping, Optional, Iterator, AsyncIterator

# --- LangChain Imports: Changed for Chat Model ---
from langchain_core.callbacks.manager import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from oauth2_token_manager import Oauth2TokenManager

def _convert_message_to_dict(message: BaseMessage) -> dict:
    """将 LangChain 的 BaseMessage 对象转换为 API 需要的字典格式。"""
    if isinstance(message, AIMessage):
        role = "assistant"
    elif isinstance(message, HumanMessage):
        role = "user"
    else: # SystemMessage, etc.
        role = "system"
    return {"role": role, "content": message.content}

class CustomOauthChatModel(BaseChatModel):
    """
    一个支持 OAuth2 和流式响应的自定义 LangChain 聊天模型。
    """
    # --- 从 .env 加载的配置 ---
    api_url: str
    model: str
    temperature: float

    # --- 核心组件 ---
    token_manager: Oauth2TokenManager
    
    # --- 可选：用于异步操作的客户端 ---
    async_client: httpx.AsyncClient = httpx.AsyncClient()

    @classmethod
    def from_env(cls, token_manager: Oauth2TokenManager) -> "CustomOauthChatModel":
        """通过环境变量和 Token Manager 实例化聊天模型。"""
        api_url = os.getenv("LLM_API_URL")
        model = os.getenv("LLM_MODEL_NAME")
        temperature = float(os.getenv("LLM_TEMPERATURE", 0.7))

        if not all([api_url, model]):
            raise ValueError("错误: 请确保 .env 文件中包含 LLM_API_URL 和 LLM_MODEL_NAME。")

        return cls(
            api_url=api_url,
            model=model,
            temperature=temperature,
            token_manager=token_manager
        )

    @property
    def _llm_type(self) -> str:
        return "custom_oauth_chat_model"

    def _build_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.token_manager.get_access_token()}",
            "Content-Type": "application/json", "Accept": "application/json",
        }

    def _build_payload(self, messages: List[BaseMessage], stream: bool, **kwargs: Any) -> dict:
        """根据消息列表构建请求体。"""
        payload = {
            "model": self.model,
            "messages": [_convert_message_to_dict(m) for m in messages],
            "temperature": self.temperature,
            "stream": stream,
            **kwargs
        }
        if stream:
            payload["stream_options"] = {"include_usage": False}
        return payload

    def _stream(
        self, 
        messages: List[BaseMessage], 
        stop: Optional[List[str]] = None, 
        run_manager: Optional[CallbackManagerForLLMRun] = None, 
        **kwargs: Any
    ) -> Iterator[ChatGenerationChunk]:
        headers = self._build_headers()
        payload = self._build_payload(messages, stream=True, **kwargs)

        with requests.post(self.api_url, headers=headers, json=payload, stream=True, timeout=30) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.startswith(b'data: '):
                    line_data = line[len(b'data: '):].strip()
                    if line_data == b'[DONE]': break
                    if not line_data: continue
                    try:
                        chunk_data = json.loads(line_data)
                        delta = chunk_data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if delta:
                            yield ChatGenerationChunk(message=AIMessageChunk(content=delta))
                    except json.JSONDecodeError:
                        continue

    async def _astream(
        self, 
        messages: List[BaseMessage], 
        stop: Optional[List[str]] = None, 
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None, 
        **kwargs: Any
    ) -> AsyncIterator[ChatGenerationChunk]:
        headers = self._build_headers()
        payload = self._build_payload(messages, stream=True, **kwargs)

        async with self.async_client.stream("POST", self.api_url, headers=headers, json=payload, timeout=30) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith('data: '):
                    line_data = line[len('data: '):].strip()
                    if line_data == '[DONE]': break
                    if not line_data: continue
                    try:
                        chunk_data = json.loads(line_data)
                        delta = chunk_data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if delta:
                            yield ChatGenerationChunk(message=AIMessageChunk(content=delta))
                    except json.JSONDecodeError:
                        continue

    def _generate(
        self, 
        messages: List[BaseMessage], 
        stop: Optional[List[str]] = None, 
        run_manager: Optional[CallbackManagerForLLMRun] = None, 
        **kwargs: Any
    ) -> ChatResult:
        full_response_content = ""
        for chunk in self._stream(messages, stop, run_manager, **kwargs):
            full_response_content += chunk.message.content
        message = AIMessage(content=full_response_content)
        return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"api_url": self.api_url, "model": self.model}

# --- 如何使用 ---
if __name__ == '__main__':
    import asyncio

    async def main():
        print("--- 开始执行自定义聊天模型调用测试 ---")
        dotenv_path = ".env"
        if not os.path.exists(dotenv_path):
            print(f"错误: {dotenv_path} 文件未找到。")
            return

        try:
            token_manager = Oauth2TokenManager(dotenv_path=dotenv_path)
            chat_model = CustomOauthChatModel.from_env(token_manager)

            # Chat Model 使用消息列表作为输入
            question_messages = [HumanMessage(content="你好，请用 Python 写一个 Hello World。")]
            
            print("\n--- 1. 测试同步流式调用 (stream) ---")
            print(f"问题: {question_messages[0].content}")
            print("模型回复 (流式): ")
            for chunk in chat_model.stream(question_messages, max_tokens=100):
                print(chunk.content, end="", flush=True)
            print("\n")

            print("--- 2. 测试同步非流式调用 (invoke) ---")
            response = chat_model.invoke(question_messages, max_tokens=100)
            print(f"问题: {question_messages[0].content}")
            print(f"模型回复 (非流式):\n{response.content}\n")

            print("--- 3. 测试异步流式调用 (astream) ---")
            print(f"问题: {question_messages[0].content}")
            print("模型回复 (异步流式): ")
            async for chunk in chat_model.astream(question_messages, max_tokens=100):
                print(chunk.content, end="", flush=True)
            print("\n")

        except Exception as e:
            print(f"\n执行过程中发生错误: {e}")

    asyncio.run(main())
    print("--- 测试执行结束 ---")