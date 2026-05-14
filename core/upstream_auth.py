from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from runtime_env import _get_runtime_env_value


def _read_configured_key(key: str) -> str:
    return _get_runtime_env_value(key, "")


def _read_codex_api_key() -> str:
    auth_path_raw = os.getenv("CODEX_AUTH_JSON", "").strip()
    auth_path = Path(auth_path_raw).expanduser() if auth_path_raw else Path.home() / ".codex" / "auth.json"
    if not auth_path.exists():
        return ""

    try:
        payload = json.loads(auth_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""

    value = payload.get("OPENAI_API_KEY")
    return value.strip() if isinstance(value, str) else ""


def resolve_upstream_bearer_token(token_manager: Any) -> str:
    api_key = _read_configured_key("LLM_API_KEY")
    if api_key:
        return api_key

    oauth_error: Optional[Exception] = None
    try:
        oauth_token = token_manager.get_access_token()
    except Exception as exc:
        oauth_error = exc
        oauth_token = ""

    if oauth_token:
        return oauth_token

    codex_api_key = _read_codex_api_key()
    if codex_api_key:
        return codex_api_key

    if oauth_error:
        raise oauth_error
    return ""


def build_upstream_headers(token_manager: Any, accept: Optional[str] = "application/json") -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {resolve_upstream_bearer_token(token_manager)}",
        "Content-Type": "application/json",
    }
    if accept:
        headers["Accept"] = accept
    return headers
