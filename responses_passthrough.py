"""
OpenAI Response API Passthrough Module

This module provides direct passthrough functionality for OpenAI's Response API.
When the backend LLM natively supports the Response API format, requests can be
forwarded directly without conversion to/from LangChain format.

This is particularly useful for Codex CLI and other tools that use the Response API.
"""

import json
from typing import Optional, Dict, Any, AsyncIterator
from fastapi import HTTPException, Header
from fastapi.responses import StreamingResponse
import httpx

from core.logger import get_logger
from core.oauth2_token_manager import OCAOauth2TokenManager, _request_with_warning_control
from core.upstream_auth import build_upstream_headers
from runtime_env import _get_runtime_env_value
from model_resolver import resolve_model_for_endpoint

logger = get_logger(__name__)


def _get_chat_model():
    """Lazy import to avoid circular dependency with api.py."""
    from api import get_chat_model
    return get_chat_model()


def _get_responses_api_url() -> Optional[str]:
    """Get the Responses API URL from environment (dynamic lookup)."""
    # Support both naming conventions (with or without 'S')
    url = _get_runtime_env_value("LLM_RESPONSES_API_URL", "")
    if not url:
        url = _get_runtime_env_value("LLM_RESPONSE_API_URL", "")
    return url or None


def is_passthrough_enabled() -> bool:
    """Check if passthrough mode is enabled (LLM_RESPONSES_API_URL is configured)."""
    return bool(_get_responses_api_url())


# Valid reasoning effort values
VALID_REASONING_EFFORTS = {"low", "medium", "high", "xhigh", "minimal", "none"}

# Minimum effort for pro models - efforts below "medium" get promoted
PRO_MODEL_MIN_EFFORT = "medium"
# Effort levels ordered from weakest to strongest
_EFFORT_ORDER = ["none", "minimal", "low", "medium", "high", "xhigh"]


def resolve_reasoning_effort(incoming_effort: Optional[str]) -> Optional[str]:
    """
    Resolve the reasoning effort for passthrough requests.

    Logic:
    1. If LLM_REASONING_STRENGTH is defined in .env and is a valid value, use it
    2. Otherwise, keep the incoming effort as-is

    Valid values: low, medium, high, xhigh, minimal, none

    Args:
        incoming_effort: The reasoning effort from the incoming request

    Returns:
        The resolved reasoning effort
    """
    llm_reasoning_strength = _get_runtime_env_value("LLM_REASONING_STRENGTH", "").lower()

    if llm_reasoning_strength and llm_reasoning_strength in VALID_REASONING_EFFORTS:
        logger.info(f"[PASSTHROUGH] Using configured LLM_REASONING_STRENGTH: {llm_reasoning_strength}")
        return llm_reasoning_strength

    # Return incoming effort unchanged
    return incoming_effort


def _is_pro_model(model_name: str) -> bool:
    """Check if the model name indicates a pro model (contains 'pro')."""
    return "pro" in model_name.lower()


def enforce_pro_model_min_reasoning(modified_body: Dict[str, Any]) -> None:
    """
    Ensure pro models have reasoning effort at least 'medium'.

    Pro models only support medium, high, and xhigh. If the resolved effort
    is below medium (e.g. low, minimal, none), promote it to medium.
    Mutates modified_body in place.
    """
    model = modified_body.get("model", "")
    if not _is_pro_model(model):
        return

    reasoning = modified_body.get("reasoning")
    if reasoning is None:
        # Pro model with no reasoning at all - add medium
        modified_body["reasoning"] = {"effort": PRO_MODEL_MIN_EFFORT, "summary": "auto"}
        logger.info(f"[PASSTHROUGH] Pro model '{model}' has no reasoning, adding effort={PRO_MODEL_MIN_EFFORT}")
        return

    if isinstance(reasoning, dict):
        effort = reasoning.get("effort", "").lower()
        min_idx = _EFFORT_ORDER.index(PRO_MODEL_MIN_EFFORT)
        if effort in _EFFORT_ORDER and _EFFORT_ORDER.index(effort) < min_idx:
            logger.info(
                f"[PASSTHROUGH] Pro model '{model}' effort '{effort}' below minimum, "
                f"promoting to '{PRO_MODEL_MIN_EFFORT}'"
            )
            modified_body["reasoning"]["effort"] = PRO_MODEL_MIN_EFFORT


def resolve_null_reasoning() -> Optional[Dict[str, str]]:
    """
    Resolve reasoning when it's null in the incoming request.

    If LLM_NON_REASONING_STRENGTH is defined in .env and is a valid value,
    return a reasoning object with that effort and summary="auto".

    Valid values: low, medium, high, xhigh, minimal, none

    Returns:
        A dict like {"effort": "<value>", "summary": "auto"} or None
    """
    llm_non_reasoning_strength = _get_runtime_env_value("LLM_NON_REASONING_STRENGTH", "").lower()

    if llm_non_reasoning_strength and llm_non_reasoning_strength in VALID_REASONING_EFFORTS:
        logger.info(f"[PASSTHROUGH] Using LLM_NON_REASONING_STRENGTH for null reasoning: {llm_non_reasoning_strength}")
        return {"effort": llm_non_reasoning_strength, "summary": "auto"}

    return None


def _get_token_manager() -> OCAOauth2TokenManager:
    """Get the token manager from the application context."""
    from api import lifespan_objects
    chat_model = lifespan_objects.get("chat_model")
    if chat_model and hasattr(chat_model, "token_manager"):
        return chat_model.token_manager
    raise HTTPException(
        status_code=500,
        detail="Token manager not available. Check server initialization."
    )


def _get_proxies_for_httpx() -> Optional[str]:
    """Get proxy URL if configured and enabled."""
    from api import lifespan_objects
    chat_model = lifespan_objects.get("chat_model")
    if chat_model and hasattr(chat_model, "token_manager"):
        tm = chat_model.token_manager
        if tm.connection_mode.value == "proxy" and tm.proxy_url:
            return tm.proxy_url
    return None


async def passthrough_stream_generator(
    request_body: Dict[str, Any],
    headers: Dict[str, str],
    response_id: str
) -> AsyncIterator[bytes]:
    """
    Stream Response API events directly from the backend LLM.

    Args:
        request_body: The original request body (will be JSON serialized)
        headers: Headers to forward (including Authorization)
        response_id: Unique response ID for logging

    Yields:
        Raw SSE bytes from the backend without line reassembly
    """
    api_url = _get_responses_api_url()
    if not api_url:
        raise HTTPException(
            status_code=500,
            detail="LLM_RESPONSES_API_URL not configured"
        )

    tm = _get_token_manager()
    proxy_url = _get_proxies_for_httpx()

    forward_headers = build_upstream_headers(
        tm,
        accept="text/event-stream" if request_body.get("stream") else "application/json",
    )

    timeout = float(_get_runtime_env_value("LLM_REQUEST_TIMEOUT", "180"))
    ca_bundle = _get_runtime_env_value("SSL_CERT_FILE", "") or _get_runtime_env_value("REQUESTS_CA_BUNDLE", "")
    disable_ssl = _get_runtime_env_value("DISABLE_SSL_VERIFY", "false").lower() == "true"

    import json as _json
    _body_size = len(_json.dumps(request_body))
    logger.info(
        f"[PASSTHROUGH] Starting streaming request to {api_url}, "
        f"model={request_body.get('model')}, response_id={response_id}, "
        f"request_body_size={_body_size}"
    )

    try:
        # Configure transport with proxy if needed
        transport = None
        verify = not disable_ssl
        if disable_ssl:
            verify = False
        elif ca_bundle:
            verify = ca_bundle

        if proxy_url:
            transport = httpx.AsyncHTTPTransport(
                proxy=proxy_url,
                verify=verify,
                trust_env=False,
            )

        async with httpx.AsyncClient(
            transport=transport,
            verify=verify if not transport else False,
            trust_env=False,
            timeout=timeout,
        ) as client:
            async with client.stream(
                "POST",
                api_url,
                content=json.dumps(request_body),
                headers=forward_headers,
            ) as response:
                if response.status_code >= 400:
                    error_body = ""
                    try:
                        async for chunk in response.aiter_bytes():
                            error_body += chunk.decode("utf-8", errors="replace")
                        logger.error(
                            f"[PASSTHROUGH ERROR] Status {response.status_code}, "
                            f"URL: {api_url}, Body: {error_body}"
                        )
                    except Exception as e:
                        logger.error(
                            f"[PASSTHROUGH ERROR] Status {response.status_code}, "
                            f"URL: {api_url}, Failed to read body: {e}"
                        )
                    raise HTTPException(
                        status_code=response.status_code,
                        detail={
                            "type": "error",
                            "error": {
                                "type": "api_error",
                                "message": f"Backend API error: {error_body[:500]}"
                            }
                        }
                    )

                # Proxy the raw bytes so Unicode line-separator characters inside
                # JSON payloads are not rewritten into actual newlines.
                total_response_size = 0
                async for chunk in response.aiter_bytes():
                    total_response_size += len(chunk)
                    yield chunk

                logger.info(
                    f"[PASSTHROUGH] Completed streaming response {response_id}, "
                    f"response_size={total_response_size}"
                )

    except httpx.RequestError as e:
        logger.exception(f"[PASSTHROUGH ERROR] Connection failed: {e}")
        raise HTTPException(
            status_code=502,
            detail={
                "type": "error",
                "error": {
                    "type": "connection_error",
                    "message": f"Failed to connect to backend: {str(e)}"
                }
            }
        )


async def passthrough_non_streaming(
    request_body: Dict[str, Any],
    headers: Dict[str, str],
    response_id: str
) -> Dict[str, Any]:
    """
    Forward a non-streaming Response API request to the backend LLM.

    Args:
        request_body: The original request body (will be JSON serialized)
        headers: Headers to forward (including Authorization)
        response_id: Unique response ID for logging

    Returns:
        The response from the backend as a dictionary
    """
    import requests

    api_url = _get_responses_api_url()
    if not api_url:
        raise HTTPException(
            status_code=500,
            detail="LLM_RESPONSES_API_URL not configured"
        )

    tm = _get_token_manager()

    forward_headers = build_upstream_headers(tm, accept="application/json")

    timeout = float(_get_runtime_env_value("LLM_REQUEST_TIMEOUT", "180"))
    ca_bundle = _get_runtime_env_value("SSL_CERT_FILE", "") or _get_runtime_env_value("REQUESTS_CA_BUNDLE", "")
    disable_ssl = _get_runtime_env_value("DISABLE_SSL_VERIFY", "false").lower() == "true"

    import json as _json
    _body_size = len(_json.dumps(request_body))
    logger.info(
        f"[PASSTHROUGH] Starting non-streaming request to {api_url}, "
        f"model={request_body.get('model')}, response_id={response_id}, "
        f"request_body_size={_body_size}"
    )

    # Configure proxies
    proxies = None
    if tm.connection_mode.value == "proxy" and tm.proxy_url:
        proxies = {"http": tm.proxy_url, "https": tm.proxy_url}

    verify = False if (disable_ssl or proxies) else (ca_bundle or True)

    try:
        response = _request_with_warning_control(
            "POST",
            api_url,
            verify=verify,
            data=json.dumps(request_body),
            headers=forward_headers,
            timeout=timeout,
            proxies=proxies,
        )

        if response.status_code >= 400:
            logger.error(
                f"[PASSTHROUGH ERROR] Status {response.status_code}, "
                f"URL: {api_url}, Body: {response.text[:500]}"
            )
            raise HTTPException(
                status_code=response.status_code,
                detail={
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": f"Backend API error: {response.text[:500]}"
                    }
                }
            )

        response_size = len(response.text)
        logger.info(
            f"[PASSTHROUGH] Completed non-streaming response {response_id}, "
            f"response_size={response_size}"
        )
        return response.json()

    except requests.exceptions.RequestException as e:
        logger.exception(f"[PASSTHROUGH ERROR] Connection failed: {e}")
        raise HTTPException(
            status_code=502,
            detail={
                "type": "error",
                "error": {
                    "type": "connection_error",
                    "message": f"Failed to connect to backend: {str(e)}"
                }
            }
        )


async def create_response_passthrough(
    request_body: Dict[str, Any],
    authorization: Optional[str] = Header(None)
):
    """
    Handle a Response API request by passing it through to the backend LLM.

    This function is used when LLM_RESPONSES_API_URL is configured and the
    backend LLM natively supports the Response API format.

    Args:
        request_body: The raw request body as a dictionary
        authorization: Optional Authorization header from client

    Returns:
        StreamingResponse (streaming) or dict (non-streaming)
    """
    import uuid
    import copy

    # Generate a response ID for logging/tracking
    response_id = f"resp_{uuid.uuid4().hex[:24]}"

    # Resolve the model name before forwarding
    original_model = request_body.get("model", "")
    try:
        chat_model = _get_chat_model()
        resolved_model = resolve_model_for_endpoint(
            original_model,
            "LLM_RESPONSES_MODEL_NAME",
            "RESPONSES",
            chat_model.model_api_support,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=500,
            detail={"type": "error", "error": {"type": "configuration_error", "message": str(e)}},
        )

    # Create a copy of the request body with resolved model
    modified_body = copy.deepcopy(request_body)
    modified_body["model"] = resolved_model

    # Extract original reasoning effort for logging
    original_reasoning = request_body.get("reasoning")
    original_effort = "none"
    if isinstance(original_reasoning, dict):
        original_effort = original_reasoning.get("effort", "none")
        if original_effort:
            original_effort = original_effort.capitalize()
        else:
            original_effort = "none"

    # Resolve reasoning effort
    reasoning_value = modified_body.get("reasoning")
    if reasoning_value is None:
        # Handle missing or null reasoning - try to fill with LLM_NON_REASONING_STRENGTH
        resolved_reasoning = resolve_null_reasoning()
        if resolved_reasoning:
            modified_body["reasoning"] = resolved_reasoning
            logger.info(f"[PASSTHROUGH] Null/missing reasoning replaced with: {resolved_reasoning}")
    elif isinstance(reasoning_value, dict):
        # Handle existing reasoning dict - override effort if configured
        incoming_effort = reasoning_value.get("effort")
        resolved_effort = resolve_reasoning_effort(incoming_effort)
        if resolved_effort != incoming_effort:
            modified_body["reasoning"]["effort"] = resolved_effort
            logger.info(f"[PASSTHROUGH] Reasoning effort resolved: {incoming_effort} -> {resolved_effort}")

    # Enforce minimum reasoning effort for pro models
    enforce_pro_model_min_reasoning(modified_body)

    # Extract final reasoning effort for logging
    final_reasoning = modified_body.get("reasoning")
    final_effort = "none"
    if isinstance(final_reasoning, dict):
        final_effort = final_reasoning.get("effort", "none")
        if final_effort:
            final_effort = final_effort.capitalize()
        else:
            final_effort = "none"

    # Log complete model resolution (only if model was resolved)
    if original_model != resolved_model:
        logger.info(f"[PASSTHROUGH] Model resolved: {original_model} ({original_effort}) -> {resolved_model} ({final_effort})")

    is_streaming = modified_body.get("stream", False)

    if is_streaming:
        return StreamingResponse(
            passthrough_stream_generator(
                request_body=modified_body,
                headers={},  # We'll use our own auth
                response_id=response_id
            ),
            media_type="text/event-stream"
        )
    else:
        return await passthrough_non_streaming(
            request_body=modified_body,
            headers={},
            response_id=response_id
        )
