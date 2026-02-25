"""
OpenAI Response API Passthrough Module

This module provides direct passthrough functionality for OpenAI's Response API.
When the backend LLM natively supports the Response API format, requests can be
forwarded directly without conversion to/from LangChain format.

This is particularly useful for Codex CLI and other tools that use the Response API.
"""

import json
import os
from typing import Optional, Dict, Any, AsyncIterator
from fastapi import HTTPException, Header
from fastapi.responses import StreamingResponse
import httpx

from dotenv import load_dotenv
from core.logger import get_logger
from core.oauth2_token_manager import OCAOauth2TokenManager

# Path to .env file for dynamic reloading
_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")

logger = get_logger(__name__)


def _reload_env() -> None:
    """Reload .env file to pick up runtime changes."""
    load_dotenv(_ENV_PATH, override=True)


def _get_responses_api_url() -> Optional[str]:
    """Get the Responses API URL from environment (dynamic lookup)."""
    _reload_env()
    # Support both naming conventions (with or without 'S')
    url = os.getenv("LLM_RESPONSES_API_URL", "").strip()
    if not url:
        url = os.getenv("LLM_RESPONSE_API_URL", "").strip()
    return url or None


def is_passthrough_enabled() -> bool:
    """Check if passthrough mode is enabled (LLM_RESPONSES_API_URL is configured)."""
    return bool(_get_responses_api_url())


def resolve_passthrough_model(incoming_model: Optional[str]) -> str:
    """
    Resolve the model name for passthrough requests.

    Logic:
    1. If LLM_MODEL_NAME is defined in .env and starts with "oca/", use it
    2. Otherwise, prefix "oca/" to the incoming model name

    Args:
        incoming_model: The model name from the incoming request (e.g., "gpt-5.1-codex-mini")

    Returns:
        The resolved model name (e.g., "oca/gpt-5.1-codex-mini" or the configured LLM_MODEL_NAME)
    """
    _reload_env()
    # Check if LLM_MODEL_NAME is configured and has oca/ prefix
    llm_model_name = os.getenv("LLM_MODEL_NAME", "").strip()
    if llm_model_name and llm_model_name.lower().startswith("oca/"):
        logger.info(f"[PASSTHROUGH] Using configured LLM_MODEL_NAME: {llm_model_name}")
        return llm_model_name

    # Otherwise, prefix oca/ to the incoming model
    if incoming_model:
        # If already has oca/ prefix, use as-is
        if incoming_model.strip().lower().startswith("oca/"):
            return incoming_model.strip()
        resolved = f"oca/{incoming_model.strip()}"
        logger.info(f"[PASSTHROUGH] Prefixed model name: {incoming_model} -> {resolved}")
        return resolved

    # Fallback to LLM_MODEL_NAME even without oca/ prefix, or empty string
    return llm_model_name or ""


# Valid reasoning effort values
VALID_REASONING_EFFORTS = {"low", "medium", "high", "xhigh", "minimal", "none"}


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
    _reload_env()
    llm_reasoning_strength = os.getenv("LLM_REASONING_STRENGTH", "").strip().lower()

    if llm_reasoning_strength and llm_reasoning_strength in VALID_REASONING_EFFORTS:
        logger.info(f"[PASSTHROUGH] Using configured LLM_REASONING_STRENGTH: {llm_reasoning_strength}")
        return llm_reasoning_strength

    # Return incoming effort unchanged
    return incoming_effort


def resolve_null_reasoning() -> Optional[Dict[str, str]]:
    """
    Resolve reasoning when it's null in the incoming request.

    If LLM_NON_REASONING_STRENGTH is defined in .env and is a valid value,
    return a reasoning object with that effort and summary="auto".

    Valid values: low, medium, high, xhigh, minimal, none

    Returns:
        A dict like {"effort": "<value>", "summary": "auto"} or None
    """
    _reload_env()
    llm_non_reasoning_strength = os.getenv("LLM_NON_REASONING_STRENGTH", "").strip().lower()

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
) -> AsyncIterator[str]:
    """
    Stream Response API events directly from the backend LLM.

    Args:
        request_body: The original request body (will be JSON serialized)
        headers: Headers to forward (including Authorization)
        response_id: Unique response ID for logging

    Yields:
        Server-Sent Events (SSE) formatted strings from the backend
    """
    api_url = _get_responses_api_url()
    if not api_url:
        raise HTTPException(
            status_code=500,
            detail="LLM_RESPONSES_API_URL not configured"
        )

    tm = _get_token_manager()
    proxy_url = _get_proxies_for_httpx()

    # Prepare headers - use the access token from token manager
    access_token = tm.get_access_token()
    forward_headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if request_body.get("stream") else "application/json",
        "Authorization": f"Bearer {access_token}",
    }

    timeout = float(os.getenv("LLM_REQUEST_TIMEOUT", "180"))
    ca_bundle = os.getenv("SSL_CERT_FILE") or os.getenv("REQUESTS_CA_BUNDLE")
    disable_ssl = os.getenv("DISABLE_SSL_VERIFY", "false").lower() == "true"

    logger.info(
        f"[PASSTHROUGH] Starting streaming request to {api_url}, "
        f"model={request_body.get('model')}, response_id={response_id}"
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

                # Stream the response back
                async for line in response.aiter_lines():
                    yield f"{line}\n"

                logger.info(f"[PASSTHROUGH] Completed streaming response {response_id}")

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

    # Prepare headers - use the access token from token manager
    access_token = tm.get_access_token()
    forward_headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {access_token}",
    }

    timeout = float(os.getenv("LLM_REQUEST_TIMEOUT", "180"))
    ca_bundle = os.getenv("SSL_CERT_FILE") or os.getenv("REQUESTS_CA_BUNDLE")
    disable_ssl = os.getenv("DISABLE_SSL_VERIFY", "false").lower() == "true"

    logger.info(
        f"[PASSTHROUGH] Starting non-streaming request to {api_url}, "
        f"model={request_body.get('model')}, response_id={response_id}"
    )

    # Configure proxies
    proxies = None
    if tm.connection_mode.value == "proxy" and tm.proxy_url:
        proxies = {"http": tm.proxy_url, "https": tm.proxy_url}

    verify = False if (disable_ssl or proxies) else (ca_bundle or True)

    try:
        response = requests.request(
            "POST",
            api_url,
            data=json.dumps(request_body),
            headers=forward_headers,
            timeout=timeout,
            proxies=proxies,
            verify=verify,
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

        logger.info(f"[PASSTHROUGH] Completed non-streaming response {response_id}")
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
    resolved_model = resolve_passthrough_model(original_model)

    # Create a copy of the request body with resolved model
    modified_body = copy.deepcopy(request_body)
    modified_body["model"] = resolved_model

    if original_model != resolved_model:
        logger.info(f"[PASSTHROUGH] Model resolved: {original_model} -> {resolved_model}")

    # Resolve reasoning effort
    if "reasoning" in modified_body:
        if modified_body["reasoning"] is None:
            # Handle null reasoning - try to fill with LLM_NON_REASONING_STRENGTH
            resolved_reasoning = resolve_null_reasoning()
            if resolved_reasoning:
                modified_body["reasoning"] = resolved_reasoning
                logger.info(f"[PASSTHROUGH] Null reasoning replaced with: {resolved_reasoning}")
        elif isinstance(modified_body["reasoning"], dict):
            # Handle existing reasoning dict - override effort if configured
            original_effort = modified_body["reasoning"].get("effort")
            resolved_effort = resolve_reasoning_effort(original_effort)
            if resolved_effort != original_effort:
                modified_body["reasoning"]["effort"] = resolved_effort
                logger.info(f"[PASSTHROUGH] Reasoning effort resolved: {original_effort} -> {resolved_effort}")

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
