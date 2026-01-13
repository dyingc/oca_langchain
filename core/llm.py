from __future__ import annotations
import os
import json
import time
import requests
import httpx
from httpx import AsyncHTTPTransport, Proxy
from typing import Any, List, Mapping, Optional, Iterator, AsyncIterator

# --- LangChain Imports ---
from langchain_core.callbacks.manager import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

# --- Local Imports ---
from .oauth2_token_manager import OCAOauth2TokenManager, ConnectionMode
import logging
from .logger import get_logger

logger = get_logger(__name__)

def _redact_headers(h: dict) -> dict:
    try:
        redacted = dict(h or {})
        if "Authorization" in redacted:
            redacted["Authorization"] = "<redacted>"
        return redacted
    except Exception:
        return {"<headers>": "<unavailable>"}

def _convert_message_to_dict(message: BaseMessage) -> dict:
    """Convert a LangChain BaseMessage object to the dictionary format needed by the API."""
    role_map = {AIMessage: "assistant", HumanMessage: "user"}
    role = role_map.get(type(message), "system")
    return {"role": role, "content": message.content}

class OCAChatModel(BaseChatModel):
    """
    A custom LangChain chat model supporting OAuth2, streaming responses, and dynamic model retrieval.
    """
    api_url: str
    model: str
    temperature: float
    models_api_url: Optional[str] = None
    llm_request_timeout: float = 120.0
    _debug: bool = False

    token_manager: OCAOauth2TokenManager
    available_models: List[str] = []

    def __init__(self, **data: Any):
        """
        Initialize model and fetch available models list.
        """
        super().__init__(**data)
        self.token_manager._debug = self._debug
        self.fetch_available_models()

        if not self.model and self.available_models:
            self.model = self.available_models[0]
            if self._debug:
                print(f"No model specified, using default model: {self.model}")

        if self.model and self.model not in self.available_models:
            raise ValueError(
                f"Error: The specified model '{self.model}' is not in the list of available models.\n"
                f"Available models: {', '.join(self.available_models)}"
            )

    def fetch_available_models(self):
        """
        Fetch and populate the available models list via API, with built-in retry logic for "cold start" issues.
        """
        if not self.models_api_url:
            if self._debug:
                print("Warning: LLM_MODELS_API_URL is not configured, cannot fetch models dynamically.")
            if self.model:
                self.available_models = [self.model]
            return

        headers = {
            "Authorization": f"Bearer {self.token_manager.get_access_token()}",
            "Accept": "application/json",
        }

        max_retries = 3
        retry_delay = 3  # seconds

        for attempt in range(max_retries):
            try:
                if self._debug:
                    print(f"Fetching available models from {self.models_api_url} (attempt {attempt + 1}/{max_retries})...")

                response = self.token_manager.request(
                    method="GET",
                    url=self.models_api_url,
                    headers=headers,
                    _do_retry=True
                )
                response.raise_for_status()

                models_data = response.json().get("data", [])
                self.available_models = [model.get("id") for model in models_data if model.get("id")]

                if not self.available_models:
                    if self._debug: print("Warning: The API returned an empty models list.")
                else:
                    if self._debug: print(f"Successfully retrieved {len(self.available_models)} available models.")

                return  # Success, exit the function

            except (ConnectionError, httpx.ConnectError, httpx.ReadTimeout) as e:
                if attempt < max_retries - 1:
                    if self._debug:
                        print(f"Failed to connect to models API. Retrying in {retry_delay} seconds... Reason: {e}")
                    time.sleep(retry_delay)
                else:
                    if self._debug:
                        print(f"Error: Unable to connect to models API after multiple attempts. Falling back to .env specified model. Reason: {e}")
                    if self.model: self.available_models = [self.model]
                    else: self.available_models = []

            except json.JSONDecodeError:
                if self._debug: print("Error: Failed to parse models API response; not a valid JSON format.")
                self.available_models = []
                return

    @classmethod
    def from_env(cls, token_manager: OCAOauth2TokenManager, debug: bool = False) -> OCAChatModel:
        api_url = os.getenv("LLM_API_URL")
        model = os.getenv("LLM_MODEL_NAME", "")
        temperature = float(os.getenv("LLM_TEMPERATURE", 0.7))
        models_api_url = os.getenv("LLM_MODELS_API_URL")
        llm_request_timeout = float(os.getenv("LLM_REQUEST_TIMEOUT", 120.0))

        if not api_url: raise ValueError("Error: Please ensure .env contains LLM_API_URL.")
        if not models_api_url and not model: raise ValueError("Error: Either LLM_MODELS_API_URL must be set or LLM_MODEL_NAME must be provided in .env.")

        return cls(
            api_url=api_url, model=model, temperature=temperature,
            token_manager=token_manager, models_api_url=models_api_url,
            llm_request_timeout=llm_request_timeout, _debug=debug
        )

    @property
    def _llm_type(self) -> str: return "oca_chat_model"

    def _build_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.token_manager.get_access_token()}",
            "Content-Type": "application/json", "Accept": "application/json",
        }

    def _build_payload(self, messages: List[BaseMessage], stream: bool, **kwargs: Any) -> dict:
        payload = {
            "model": self.model,
            "messages": [_convert_message_to_dict(m) for m in messages],
            "temperature": self.temperature,
            "stream": stream,
        }
        # Optional args passthrough
        if "max_tokens" in kwargs and kwargs["max_tokens"] is not None:
            payload["max_tokens"] = kwargs["max_tokens"]
        if "tool_choice" in kwargs and kwargs["tool_choice"] is not None:
            payload["tool_choice"] = kwargs["tool_choice"]
        if "tools" in kwargs and kwargs["tools"]:
            payload["tools"] = kwargs["tools"]
        if stream:
            payload["stream_options"] = {"include_usage": False}
        return payload

    def _stream(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        headers = self._build_headers()
        payload = self._build_payload(messages, stream=True, **kwargs)
        # Logging request
        try:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("[LLM REQUEST] headers=%s payload=%s", json.dumps(_redact_headers(headers), ensure_ascii=False), json.dumps(payload, ensure_ascii=False))
            else:
                logger.info("[LLM REQUEST] %s", json.dumps(payload, ensure_ascii=False))
        except Exception:
            pass
        try:
            response = self.token_manager.request(
                method="POST", url=self.api_url, headers=headers, json=payload,
                stream=True, _do_retry=False, request_timeout=self.llm_request_timeout
            )
            response.raise_for_status()
            try:
                self._last_response_headers = dict(response.headers)
            except Exception:
                self._last_response_headers = None
        except ConnectionError as e:
            print(f"Streaming API request failed: {e}. Retry is disabled.")
            raise

        for line in response.iter_lines():
            if line.startswith(b'data: '):
                line_data = line[len(b'data: '):].strip()
                if line_data == b'[DONE]': break
                if not line_data: continue
                try:
                    chunk_data = json.loads(line_data)
                    delta_obj = chunk_data.get("choices", [{}])[0].get("delta", {})
                    content_delta = delta_obj.get("content", "")
                    tool_calls_delta = delta_obj.get("tool_calls")
                    # Normalize legacy function_call delta into tool_calls format
                    function_call_delta = delta_obj.get("function_call")
                    if function_call_delta is not None:
                        fc_tool = {
                            "index": 0,
                            "id": function_call_delta.get("id"),
                            "type": "function",
                            "function": {
                                "name": function_call_delta.get("name"),
                                "arguments": function_call_delta.get("arguments"),
                            },
                        }
                        if tool_calls_delta is None:
                            tool_calls_delta = [fc_tool]
                        else:
                            tool_calls_delta = list(tool_calls_delta) + [fc_tool]
                    additional_kwargs = {}
                    if tool_calls_delta is not None:
                        additional_kwargs["tool_calls"] = tool_calls_delta
                    if content_delta or additional_kwargs:
                        yield ChatGenerationChunk(message=AIMessageChunk(content=content_delta or "", additional_kwargs=additional_kwargs))
                except json.JSONDecodeError: continue

    async def _astream(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Optional[AsyncCallbackManagerForLLMRun] = None, **kwargs: Any) -> AsyncIterator[ChatGenerationChunk]:
        headers = self._build_headers()
        payload = self._build_payload(messages, stream=True, **kwargs)
        # Logging request
        try:
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug("[LLM REQUEST] headers=%s payload=%s", json.dumps(_redact_headers(headers), ensure_ascii=False), json.dumps(payload, ensure_ascii=False))
            else:
                self.logger.info("[LLM REQUEST] %s", json.dumps(payload, ensure_ascii=False))
        except Exception:
            pass
        full_async_content = ""
        response_headers: dict = {}
        tool_builders_async: dict = {}
        order_async: List[Any] = []
        try:
            async for line in self.token_manager.async_stream_request(
                method="POST", url=self.api_url, headers=headers, json=payload,
                _do_retry=False, request_timeout=self.llm_request_timeout,
                on_open=lambda resp: response_headers.update(dict(resp.headers))
            ):
                if line.startswith('data: '):
                    line_data = line[len('data: '):].strip()
                    if line_data == '[DONE]': break
                    if not line_data: continue
                    try:
                        chunk_data = json.loads(line_data)
                        delta_obj = chunk_data.get("choices", [{}])[0].get("delta", {})
                        content_delta = delta_obj.get("content", "")
                        tool_calls_delta = delta_obj.get("tool_calls")
                        # Normalize legacy function_call delta into tool_calls format
                        function_call_delta = delta_obj.get("function_call")
                        if function_call_delta is not None:
                            fc_tool = {
                                "index": 0,
                                "id": function_call_delta.get("id"),
                                "type": "function",
                                "function": {
                                    "name": function_call_delta.get("name"),
                                    "arguments": function_call_delta.get("arguments"),
                                },
                            }
                            if tool_calls_delta is None:
                                tool_calls_delta = [fc_tool]
                            else:
                                tool_calls_delta = list(tool_calls_delta) + [fc_tool]
                        additional_kwargs = {}
                        if tool_calls_delta is not None:
                            additional_kwargs["tool_calls"] = tool_calls_delta
                        # Accumulate tool_calls into builders for final logging
                        try:
                            tcs = tool_calls_delta
                            if isinstance(tcs, list):
                                for tc in tcs:
                                    idx = tc.get("index")
                                    tid = tc.get("id")
                                    if idx is not None:
                                        key = ("i", idx)
                                    elif tid is not None:
                                        key = ("id", tid)
                                    else:
                                        key = ("i", 0)
                                    if key not in tool_builders_async:
                                        tool_builders_async[key] = {"type": "function", "id": tid, "function": {"name": None, "arguments": ""}}
                                        order_async.append(key)
                                    b = tool_builders_async[key]
                                    if "type" in tc and tc["type"]:
                                        b["type"] = tc["type"]
                                    if tid and not b.get("id"):
                                        b["id"] = tid
                                    fdelta = tc.get("function") or {}
                                    if "name" in fdelta and fdelta["name"]:
                                        b["function"]["name"] = fdelta["name"]
                                    if "arguments" in fdelta and fdelta["arguments"]:
                                        b["function"]["arguments"] += fdelta["arguments"]
                        except Exception:
                            pass
                        if content_delta or additional_kwargs:
                            if content_delta:
                                full_async_content += content_delta
                            yield ChatGenerationChunk(message=AIMessageChunk(content=content_delta or "", additional_kwargs=additional_kwargs))
                    except json.JSONDecodeError: continue
            # After streaming completes, build final tool_calls and log final response
            final_tool_calls_async = None
            if order_async:
                final_tool_calls_async = []
                for key in order_async:
                    b = tool_builders_async[key]
                    if "function" not in b or b["function"] is None:
                        b["function"] = {"name": None, "arguments": ""}
                    if "arguments" not in b["function"] or b["function"]["arguments"] is None:
                        b["function"]["arguments"] = ""
                    if not isinstance(b["function"]["arguments"], str):
                        b["function"]["arguments"] = str(b["function"]["arguments"])
                    final_tool_calls_async.append(b)
            try:
                log_obj = {"content": full_async_content, "tool_calls": final_tool_calls_async or []}
                if logger.isEnabledFor(logging.DEBUG):
                    if response_headers:
                        logger.debug("[LLM RESPONSE] headers=%s body=%s", json.dumps(response_headers, ensure_ascii=False), json.dumps(log_obj, ensure_ascii=False))
                    else:
                        logger.debug("[LLM RESPONSE] body=%s", json.dumps(log_obj, ensure_ascii=False))
                else:
                    logger.info("[LLM RESPONSE] %s", json.dumps(log_obj, ensure_ascii=False))
            except Exception:
                pass
        except ConnectionError as e:
            print(f"Async streaming API request failed: {e}. Retry is disabled.")
            raise

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> ChatResult:
        # Aggregate content and reconstruct streaming tool_calls deltas into a final OpenAI-compatible list
        full_response_content = ""
        tool_builders: dict = {}
        order: List[Any] = []
        for chunk in self._stream(messages, stop, run_manager, **kwargs):
            # Accumulate text content
            if getattr(chunk.message, "content", None):
                full_response_content += chunk.message.content
            # Accumulate tool_calls deltas
            try:
                additional = getattr(chunk.message, "additional_kwargs", {}) or {}
                tcs = additional.get("tool_calls")
                if isinstance(tcs, list):
                    for tc in tcs:
                        # OpenAI streaming provides an index; fall back to id or 0
                        idx = tc.get("index")
                        tid = tc.get("id")
                        if idx is not None:
                            key = ("i", idx)
                        elif tid is not None:
                            key = ("id", tid)
                        else:
                            key = ("i", 0)
                        if key not in tool_builders:
                            tool_builders[key] = {"type": "function", "id": tid, "function": {"name": None, "arguments": ""}}
                            order.append(key)
                        b = tool_builders[key]
                        # Merge fields
                        if "type" in tc and tc["type"]:
                            b["type"] = tc["type"]
                        if tid and not b.get("id"):
                            b["id"] = tid
                        fdelta = tc.get("function") or {}
                        if "name" in fdelta and fdelta["name"]:
                            b["function"]["name"] = fdelta["name"]
                        if "arguments" in fdelta and fdelta["arguments"]:
                            # Append incremental argument chunks
                            b["function"]["arguments"] += fdelta["arguments"]
            except Exception:
                pass
        # Build final tool_calls list if any
        final_tool_calls = None
        if order:
            final_tool_calls = []
            for key in order:
                b = tool_builders[key]
                # Ensure required structure
                if "function" not in b or b["function"] is None:
                    b["function"] = {"name": None, "arguments": ""}
                if "arguments" not in b["function"] or b["function"]["arguments"] is None:
                    b["function"]["arguments"] = ""
                # Coerce to str to satisfy OpenAI schema
                if not isinstance(b["function"]["arguments"], str):
                    b["function"]["arguments"] = str(b["function"]["arguments"])
                final_tool_calls.append(b)
        # Log final response
        try:
            headers_to_log = getattr(self, "_last_response_headers", None)
            log_obj = {"content": full_response_content, "tool_calls": final_tool_calls or []}
            if logger.isEnabledFor(logging.DEBUG):
                if headers_to_log is not None:
                    logger.debug("[LLM RESPONSE] headers=%s body=%s", json.dumps(headers_to_log, ensure_ascii=False), json.dumps(log_obj, ensure_ascii=False))
                else:
                    logger.debug("[LLM RESPONSE] body=%s", json.dumps(log_obj, ensure_ascii=False))
            else:
                logger.info("[LLM RESPONSE] %s", json.dumps(log_obj, ensure_ascii=False))
        except Exception:
            pass
        if final_tool_calls is not None:
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=full_response_content, additional_kwargs={"tool_calls": final_tool_calls}))])
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=full_response_content))])

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"api_url": self.api_url, "model": self.model, "models_api_url": self.models_api_url}

if __name__ == '__main__':
    import asyncio
    import yaml
    debug_mode = os.getenv("DEBUG_MODE", "False").lower() == "true"
    async def main():
        if debug_mode: print("--- Starting custom chat model call test ---")
        dotenv_path, config_path = ".env", "config.yaml"
        if not os.path.exists(dotenv_path):
            if debug_mode: print(f"Error: {dotenv_path} file not found.")
            return
        if not os.path.exists(config_path):
            if debug_mode: print(f"Error: {config_path} file not found.")
            return
        try:
            with open(config_path, 'r') as f: config = yaml.safe_load(f)
            test_prompt = config["llm_prompts"]["test_prompt"]
            token_manager = OCAOauth2TokenManager(dotenv_path=dotenv_path, debug=debug_mode)
            chat_model = OCAChatModel.from_env(token_manager, debug=debug_mode)
            if debug_mode:
                print(f"\n--- {len(chat_model.available_models)} available models detected ---")
                for i, model_id in enumerate(chat_model.available_models): print(f"{i+1}. {model_id}")
            if "oca/gpt-4.1" in chat_model.available_models:
                chat_model.model = "oca/gpt-4.1"
                if debug_mode: print(f"--- Switched to model: {chat_model.model} ---\n")
            else:
                if debug_mode: print(f"--- Using model: {chat_model.model} ---\n")
            if not chat_model.available_models:
                if debug_mode: print("Error: No available models, cannot perform call test.")
                return
            question_messages = [HumanMessage(content=test_prompt)]
            if debug_mode:
                print("\n--- 1. Testing sync streaming call (stream) ---")
                print(f"Question: {question_messages[0].content[:100]}...")
                print("Model response (stream): ")
            for chunk in chat_model.stream(question_messages, max_tokens=100): print(chunk.content, end="", flush=True)
            if debug_mode: print("\n")
            response = chat_model.invoke(question_messages, max_tokens=100)
            if debug_mode:
                print("--- 2. Testing sync non-streaming call (invoke) ---")
                print(f"Question: {question_messages[0].content[:100]}...")
                print(f"Model response (non-stream):\n{response.content}\n")
            if debug_mode:
                print("--- 3. Testing async streaming call (astream) ---")
                print(f"Question: {question_messages[0].content[:100]}...")
                print("Model response (async stream): ")
            async for chunk in chat_model.astream(question_messages, max_tokens=100): print(chunk.content, end="", flush=True)
            if debug_mode: print("\n")
        except Exception as e:
            if debug_mode: print(f"\nError occurred during test execution: {e}")
    asyncio.run(main())
    if debug_mode: print("--- Test execution finished ---")
