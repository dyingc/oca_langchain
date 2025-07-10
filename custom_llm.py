from __future__ import annotations
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
    它能动态获取可用的模型列表。
    """
    # --- 从 .env 加载的配置 ---
    api_url: str
    model: str
    temperature: float
    models_api_url: Optional[str] = None

    # --- 核心组件 ---
    token_manager: Oauth2TokenManager

    # --- 动态获取的数据 ---
    available_models: List[str] = []

    # --- 可选：用于异步操作的客户端 ---
    async_client: httpx.AsyncClient = httpx.AsyncClient()

    def __init__(self, **data: Any):
        """
        初始化模型，并获取可用模型列表。
        """
        super().__init__(**data)
        self._fetch_available_models()

        # 如果用户没有指定模型，则使用列表中的第一个作为默认模型
        if not self.model and self.available_models:
            self.model = self.available_models[0]
            print(f"未指定模型，使用默认模型: {self.model}")

        # 验证指定的模型是否有效
        if self.model and self.model not in self.available_models:
            raise ValueError(
                f"错误: 指定的模型 '{self.model}' 不在可用模型列表中。\n"
                f"可用模型: {', '.join(self.available_models)}"
            )

    def _fetch_available_models(self):
        """
        调用 API 获取并填充可用模型列表。
        """
        if not self.models_api_url:
            print("警告: 未配置 LLM_MODELS_API_URL，无法动态获取模型列表。")
            # 如果没有配置模型URL，但 .env 中有模型名称，则将其作为唯一可用模型
            if self.model:
                self.available_models = [self.model]
            return

        headers = {
            "Authorization": f"Bearer {self.token_manager.get_access_token()}",
            "Accept": "application/json",
        }
        print(f"正在从 {self.models_api_url} 获取可用模型列表...")
        try:
            response = requests.get(self.models_api_url, headers=headers, timeout=15)
            response.raise_for_status()

            models_data = response.json().get("data", [])
            # 根据您提供的响应格式，从每个字典中提取 'id'
            self.available_models = [model.get("id") for model in models_data if model.get("id")]

            if not self.available_models:
                print("警告: 从 API 获取的模型列表为空。")
            else:
                print(f"成功获取到 {len(self.available_models)} 个可用模型。")

        except requests.exceptions.RequestException as e:
            print(f"错误: 调用模型 API 时出错: {e}")
            # 出错时，如果 .env 中有模型，则退回使用该模型
            if self.model:
                self.available_models = [self.model]
                print(f"将回退到使用 .env 文件中指定的模型: {self.model}")
            else:
                 self.available_models = []
        except json.JSONDecodeError:
            print("错误: 解析模型 API 响应失败，响应不是有效的 JSON 格式。")
            self.available_models = []


    @classmethod
    def from_env(cls, token_manager: Oauth2TokenManager) -> "CustomOauthChatModel":
        """通过环境变量和 Token Manager 实例化聊天模型。"""
        api_url = os.getenv("LLM_API_URL")
        # LLM_MODEL_NAME 现在是可选的
        model = os.getenv("LLM_MODEL_NAME", "")
        temperature = float(os.getenv("LLM_TEMPERATURE", 0.7))
        models_api_url = os.getenv("LLM_MODELS_API_URL")

        if not api_url:
            raise ValueError("错误: 请确保 .env 文件中包含 LLM_API_URL。")

        # 如果没有模型获取API，则必须在.env中指定一个模型
        if not models_api_url and not model:
            raise ValueError("错误: 必须配置 LLM_MODELS_API_URL 或在 .env 中提供 LLM_MODEL_NAME。")

        return cls(
            api_url=api_url,
            model=model,
            temperature=temperature,
            token_manager=token_manager,
            models_api_url=models_api_url
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
        return {
            "api_url": self.api_url,
            "model": self.model,
            "models_api_url": self.models_api_url
        }

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
            # 1. 初始化 Token Manager 和 Chat Model
            #    模型初始化时会自动获取可用模型列表
            token_manager = Oauth2TokenManager(dotenv_path=dotenv_path)
            chat_model = CustomOauthChatModel.from_env(token_manager)

            # 2. 打印获取到的模型信息
            print(f"\n--- 检测到可用模型 ({len(chat_model.available_models)}个) ---")
            for i, model_id in enumerate(chat_model.available_models):
                print(f"{i+1}. {model_id}")
            print(f"--- 当前使用的模型: {chat_model.model} ---\n")

            # 如果没有可用模型，则无法继续
            if not chat_model.available_models:
                print("错误：没有可用的模型，无法执行调用测试。")
                return

            # 3. 准备输入消息
            question_messages = [HumanMessage(content="你好，请用 Python 写一个 Hello World。")]

            # 4. 执行各种调用测试
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
