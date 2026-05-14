import asyncio
import json
from datetime import datetime, timedelta, timezone

from core.oauth2_token_manager import OCAOauth2TokenManager
from responses_passthrough import passthrough_stream_generator


LINE_SEPARATOR = "\u2028"


class DummyTokenManager:
    def get_access_token(self):
        return "test-token"


class FailingTokenManager:
    def get_access_token(self):
        raise RuntimeError("OAuth unavailable")


class FakeStreamingResponse:
    def __init__(self, raw_chunks):
        self.status_code = 200
        self.headers = {}
        self._raw_chunks = list(raw_chunks)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self):
        return None

    async def aiter_bytes(self):
        for chunk in self._raw_chunks:
            yield chunk

    async def aiter_lines(self):
        text = b"".join(self._raw_chunks).decode("utf-8")
        for line in text.splitlines():
            yield line


def make_async_client(fake_response):
    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def stream(self, *args, **kwargs):
            return fake_response

    return FakeAsyncClient


def make_capturing_async_client(fake_response, captured):
    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def stream(self, *args, **kwargs):
            captured["headers"] = kwargs.get("headers", {})
            return fake_response

    return FakeAsyncClient


def split_bytes_inside_marker(text, marker):
    raw = text.encode("utf-8")
    marker_bytes = marker.encode("utf-8")
    start = raw.index(marker_bytes)
    return raw, [raw[: start + 1], raw[start + 1 : start + 2], raw[start + 2 :]]


async def collect_bytes(async_iterable):
    chunks = []
    async for chunk in async_iterable:
        if isinstance(chunk, str):
            chunk = chunk.encode("utf-8")
        chunks.append(bytes(chunk))
    return b"".join(chunks)


async def collect_lines(async_iterable):
    return [line async for line in async_iterable]


def test_passthrough_stream_generator_preserves_raw_sse_bytes(monkeypatch):
    raw_text = f'data: {{"delta":"{LINE_SEPARATOR}foo","obfuscation":"x"}}\n\n'
    raw_stream, raw_chunks = split_bytes_inside_marker(raw_text, LINE_SEPARATOR)
    fake_response = FakeStreamingResponse(raw_chunks)

    monkeypatch.setattr("responses_passthrough._get_responses_api_url", lambda: "https://example.test/v1/responses")
    monkeypatch.setattr("responses_passthrough._get_token_manager", lambda: DummyTokenManager())
    monkeypatch.setattr("responses_passthrough._get_proxies_for_httpx", lambda: None)
    monkeypatch.setattr("responses_passthrough._get_runtime_env_value", lambda key, default="": default)
    monkeypatch.setattr("responses_passthrough.httpx.AsyncClient", make_async_client(fake_response))

    streamed = asyncio.run(
        collect_bytes(
            passthrough_stream_generator(
                request_body={"model": "oca/test", "stream": True},
                headers={},
                response_id="resp_test",
            )
        )
    )

    assert streamed == raw_stream


def test_passthrough_stream_generator_prefers_upstream_api_key(monkeypatch, tmp_path):
    raw_text = 'data: {"type":"response.completed"}\n\n'
    fake_response = FakeStreamingResponse([raw_text.encode("utf-8")])
    captured = {}

    env_file = tmp_path / ".env"
    env_file.write_text('LLM_API_KEY="upstream-api-key"\n', encoding="utf-8")
    monkeypatch.setattr("runtime_env._ENV_PATH", str(env_file))
    monkeypatch.setattr("responses_passthrough._get_responses_api_url", lambda: "https://example.test/responses")
    monkeypatch.setattr("responses_passthrough._get_token_manager", lambda: DummyTokenManager())
    monkeypatch.setattr("responses_passthrough._get_proxies_for_httpx", lambda: None)
    monkeypatch.setattr(
        "responses_passthrough.httpx.AsyncClient",
        make_capturing_async_client(fake_response, captured),
    )

    asyncio.run(
        collect_bytes(
            passthrough_stream_generator(
                request_body={"model": "oca/test", "stream": True},
                headers={},
                response_id="resp_test",
            )
        )
    )

    assert captured["headers"]["Authorization"] == "Bearer upstream-api-key"


def test_passthrough_stream_generator_prefers_oauth_before_codex_auth(monkeypatch, tmp_path):
    raw_text = 'data: {"type":"response.completed"}\n\n'
    fake_response = FakeStreamingResponse([raw_text.encode("utf-8")])
    captured = {}

    env_file = tmp_path / ".env"
    env_file.write_text('LLM_API_KEY=""\n', encoding="utf-8")
    auth_file = tmp_path / "auth.json"
    auth_file.write_text(json.dumps({"OPENAI_API_KEY": "codex-api-key"}), encoding="utf-8")

    monkeypatch.setattr("runtime_env._ENV_PATH", str(env_file))
    monkeypatch.setenv("CODEX_AUTH_JSON", str(auth_file))
    monkeypatch.setattr("responses_passthrough._get_responses_api_url", lambda: "https://example.test/responses")
    monkeypatch.setattr("responses_passthrough._get_token_manager", lambda: DummyTokenManager())
    monkeypatch.setattr("responses_passthrough._get_proxies_for_httpx", lambda: None)
    monkeypatch.setattr(
        "responses_passthrough.httpx.AsyncClient",
        make_capturing_async_client(fake_response, captured),
    )

    asyncio.run(
        collect_bytes(
            passthrough_stream_generator(
                request_body={"model": "oca/test", "stream": True},
                headers={},
                response_id="resp_test",
            )
        )
    )

    assert captured["headers"]["Authorization"] == "Bearer test-token"


def test_passthrough_stream_generator_falls_back_to_codex_auth(monkeypatch, tmp_path):
    raw_text = 'data: {"type":"response.completed"}\n\n'
    fake_response = FakeStreamingResponse([raw_text.encode("utf-8")])
    captured = {}

    env_file = tmp_path / ".env"
    env_file.write_text('LLM_API_KEY=""\n', encoding="utf-8")
    auth_file = tmp_path / "auth.json"
    auth_file.write_text(json.dumps({"OPENAI_API_KEY": "codex-api-key"}), encoding="utf-8")

    monkeypatch.setattr("runtime_env._ENV_PATH", str(env_file))
    monkeypatch.setenv("CODEX_AUTH_JSON", str(auth_file))
    monkeypatch.setattr("responses_passthrough._get_responses_api_url", lambda: "https://example.test/responses")
    monkeypatch.setattr("responses_passthrough._get_token_manager", lambda: FailingTokenManager())
    monkeypatch.setattr("responses_passthrough._get_proxies_for_httpx", lambda: None)
    monkeypatch.setattr(
        "responses_passthrough.httpx.AsyncClient",
        make_capturing_async_client(fake_response, captured),
    )

    asyncio.run(
        collect_bytes(
            passthrough_stream_generator(
                request_body={"model": "oca/test", "stream": True},
                headers={},
                response_id="resp_test",
            )
        )
    )

    assert captured["headers"]["Authorization"] == "Bearer codex-api-key"


def test_async_stream_request_preserves_unicode_separator_inside_json(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "OAUTH_HOST=example.test",
                "OAUTH_CLIENT_ID=test-client",
                "OAUTH_ACCESS_TOKEN=test-token",
                "OAUTH_ACCESS_TOKEN_EXPIRES_AT=2999-01-01T00:00:00+00:00",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    manager = OCAOauth2TokenManager(str(env_file))
    manager.access_token = "test-token"
    manager.expires_at = datetime.now(timezone.utc) + timedelta(hours=1)

    raw_text = f'data: {{"choices":[{{"delta":{{"content":"{LINE_SEPARATOR}foo"}}}}]}}\n\n'
    _, raw_chunks = split_bytes_inside_marker(raw_text, LINE_SEPARATOR)
    fake_response = FakeStreamingResponse(raw_chunks)

    monkeypatch.delenv("FORCE_PROXY", raising=False)
    monkeypatch.setattr("core.oauth2_token_manager.httpx.AsyncClient", make_async_client(fake_response))

    lines = asyncio.run(
        collect_lines(
            manager.async_stream_request(
                method="POST",
                url="https://example.test/v1/chat/completions",
                _do_retry=False,
                headers={},
                json={},
            )
        )
    )

    assert lines == [
        f'data: {{"choices":[{{"delta":{{"content":"{LINE_SEPARATOR}foo"}}}}]}}',
        "",
    ]
