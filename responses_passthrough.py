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

# Load .env file
load_dotenv()

logger = get_logger(__name__)


def _get_responses_api_url() -> Optional[str]:
    """Get the Responses API URL from environment (dynamic lookup)."""
    # Support both naming conventions (with or without 'S')
    url = os.getenv("LLM_RESPONSES_API_URL", "").strip()
    if not url:
        url = os.getenv("LLM_RESPONSE_API_URL", "").strip()
    return url or None


def is_passthrough_enabled() -> bool:
    """Check if passthrough mode is enabled (LLM_RESPONSES_API_URL is configured)."""
    return bool(_get_responses_api_url())


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

    # Generate a response ID for logging/tracking
    response_id = f"resp_{uuid.uuid4().hex[:24]}"

    is_streaming = request_body.get("stream", False)

    if is_streaming:
        return StreamingResponse(
            passthrough_stream_generator(
                request_body=request_body,
                headers={},  # We'll use our own auth
                response_id=response_id
            ),
            media_type="text/event-stream"
        )
    else:
        return await passthrough_non_streaming(
            request_body=request_body,
            headers={},
            response_id=response_id
        )
