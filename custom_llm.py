import os
import json
import requests
import httpx
from typing import Any, List, Mapping, Optional, Iterator, AsyncIterator

from langchain_core.callbacks.manager import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk

from oauth2_token_manager import Oauth2TokenManager

class CustomOauthLLM(LLM):
    """
    一个支持 OAuth2 和流式响应的自定义 LangChain LLM。
    """
    # --- 从 .env 加载的配置 ---
    api_url: str
    model: str
    system_prompt: str
    temperature: float

    # --- 核心组件 ---
    token_manager: Oauth2TokenManager
    
    # --- 可选：用于异步操作的客户端 ---
    async_client: httpx.AsyncClient = httpx.AsyncClient()

    @classmethod
    def from_env(cls, token_manager: Oauth2TokenManager) -> "CustomOauthLLM":
        """通过环境变量和 Token Manager 实例化 LLM。"""
        api_url = os.getenv("LLM_API_URL")
        model = os.getenv("LLM_MODEL_NAME")
        system_prompt = os.getenv("LLM_SYSTEM_PROMPT", "You are a helpful assistant.")
        temperature = float(os.getenv("LLM_TEMPERATURE", 0.7))

        if not all([api_url, model]):
            raise ValueError("错误: 请确保 .env 文件中包含 LLM_API_URL 和 LLM_MODEL_NAME。")

        return cls(
            api_url=api_url,
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            token_manager=token_manager
        )

    @property
    def _llm_type(self) -> str:
        return "custom_oauth_streaming_llm"

    def _build_headers(self) -> dict:
        """构建请求头。"""
        access_token = self.token_manager.get_access_token()
        return {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _build_payload(self, prompt: str, stream: bool, **kwargs: Any) -> dict:
        """构建请求体。"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": stream,
            **kwargs
        }
        if stream:
            payload["stream_options"] = {"include_usage": False}
            
        return payload

    def _stream(
        self, 
        prompt: str, 
        stop: Optional[List[str]] = None, 
        run_manager: Optional[CallbackManagerForLLMRun] = None, 
        **kwargs: Any
    ) -> Iterator[GenerationChunk]:
        """同步流式方法。"""
        headers = self._build_headers()
        payload = self._build_payload(prompt, stream=True, **kwargs)

        with requests.post(self.api_url, headers=headers, json=payload, stream=True, timeout=30) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.startswith(b'data: '):
                    line_data = line[len(b'data: '):].strip()
                    if line_data == b'[DONE]':
                        break
                    if not line_data:
                        continue
                    try:
                        chunk_data = json.loads(line_data)
                        delta = chunk_data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if delta:
                            yield GenerationChunk(text=delta)
                    except json.JSONDecodeError:
                        print(f"警告: 无法解析的 JSON 行: {line_data}")
                        continue

    async def _astream(
        self, 
        prompt: str, 
        stop: Optional[List[str]] = None, 
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None, 
        **kwargs: Any
    ) -> AsyncIterator[GenerationChunk]:
        """异步流式方法。"""
        headers = self._build_headers()
        payload = self._build_payload(prompt, stream=True, **kwargs)

        

        async with self.async_client.stream("POST", self.api_url, headers=headers, json=payload, timeout=30) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith('data: '):
                    line_data = line[len('data: '):].strip()
                    if line_data == '[DONE]':
                        break
                    if not line_data:
                        continue
                    try:
                        chunk_data = json.loads(line_data)
                        delta = chunk_data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if delta:
                            yield GenerationChunk(text=delta)
                    except json.JSONDecodeError:
                        print(f"警告: 无法解析的 JSON 行: {line_data}")
                        continue

    def _call(
        self, 
        prompt: str, 
        stop: Optional[List[str]] = None, 
        run_manager: Optional[CallbackManagerForLLMRun] = None, 
        **kwargs: Any
    ) -> str:
        """同步非流式调用，通过聚合流式结果实现。"""
        full_response = ""
        for chunk in self._stream(prompt, stop, run_manager, **kwargs):
            full_response += chunk.text
        return full_response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"api_url": self.api_url, "model": self.model}

# --- 如何使用 ---
if __name__ == '__main__':
    import asyncio

    async def main():
        print("--- 开始执行自定义 LLM 调用测试 ---")
        dotenv_path = ".env"
        if not os.path.exists(dotenv_path):
            print(f"错误: {dotenv_path} 文件未找到。请根据模板创建并填入您的凭证。")
            return

        try:
            # 1. 初始化 Token Manager
            token_manager = Oauth2TokenManager(dotenv_path=dotenv_path)

            # 2. 使用 from_env 类方法实例化 LLM
            llm = CustomOauthLLM.from_env(token_manager)

            question = "你好，请用 Python 写一个 Hello World。"
            
            print("\n--- 1. 测试同步流式调用 (stream) ---")
            print(f"问题: {question}")
            print("模型回复 (流式): ")
            for chunk in llm.stream(question, max_tokens=100):
                print(chunk, end="", flush=True)
            print("\n")

            print("--- 2. 测试同步非流式调用 (invoke) ---")
            response = llm.invoke(question, max_tokens=100)
            print(f"问题: {question}")
            print(f"模型回复 (非流式):\n{response}\n")

            print("--- 3. 测试异步流式调用 (astream) ---")
            print(f"问题: {question}")
            print("模型回复 (异步流式): ")
            async for chunk in llm.astream(question, max_tokens=100):
                print(chunk, end="", flush=True)
            print("\n")

        except (ValueError, FileNotFoundError, requests.exceptions.RequestException, httpx.RequestError) as e:
            print(f"\n执行过程中发生错误: {e}")
            print("请检查您的 .env 文件配置和网络连接。")

    # 运行主异步函数
    asyncio.run(main())
    print("--- 测试执行结束 ---")
