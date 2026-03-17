import logging
from typing import Dict, List, Optional

from runtime_env import _get_runtime_env_value

logger = logging.getLogger(__name__)

FALLBACK_MODEL = "oca/gpt-5.4"


def _normalize_model_id(model: str) -> str:
    """Add oca/ prefix if missing; strip duplicate oca/ prefixes."""
    model = model.strip()
    # Strip all leading oca/ prefixes, then add exactly one
    while model.lower().startswith("oca/"):
        model = model[4:]
    return f"oca/{model}"


def resolve_model_for_endpoint(
    incoming_model: Optional[str],
    env_key: str,
    endpoint_type: str,
    model_api_support: Dict[str, List[str]],
) -> str:
    """Resolve which model to use for a given endpoint type.

    Args:
        incoming_model: Model name from the request (may lack oca/ prefix).
        env_key: Env var for operator override (LLM_MODEL_NAME or LLM_RESPONSES_MODEL_NAME).
        endpoint_type: Required capability (CHAT_COMPLETIONS or RESPONSES).
        model_api_support: Dict mapping model_id -> list of supported endpoint types (uppercase).
                           Empty dict means catalog was not loaded at startup (fail-open mode).

    Returns:
        Resolved model name (always non-empty string).

    Raises:
        ValueError: If env override is set and the model is misconfigured for the endpoint.
    """
    # Step 1: Normalize incoming
    incoming_stripped = (incoming_model or "").strip()
    if not incoming_stripped:
        logger.warning(
            f"[MODEL RESOLUTION] Empty incoming model for {endpoint_type}; "
            f"falling back to {FALLBACK_MODEL}"
        )
        return FALLBACK_MODEL

    # Step 2: Check env override
    raw_env = _get_runtime_env_value(env_key, "")
    if raw_env:
        normalized_env = _normalize_model_id(raw_env)

        if model_api_support:
            # Catalog available — validate the override
            if normalized_env not in model_api_support:
                raise ValueError(
                    f"Configured {env_key}='{raw_env}' (resolved to '{normalized_env}') "
                    f"is not in the model catalog. Check your .env."
                )
            supported = model_api_support[normalized_env]
            if supported and endpoint_type not in supported:
                raise ValueError(
                    f"Configured {env_key}='{raw_env}' does not support {endpoint_type}. "
                    f"Supported endpoints for this model: {supported}. Check your .env."
                )
            return normalized_env
        else:
            # Catalog unavailable — fail-open, warn
            logger.warning(
                f"[MODEL RESOLUTION] Model catalog unavailable; using configured "
                f"{env_key}='{normalized_env}' without endpoint validation"
            )
            return normalized_env

    # Step 3: Normalize incoming with oca/ prefix
    candidate = _normalize_model_id(incoming_stripped)

    # Step 4: Check catalog
    if not model_api_support:
        # Catalog unavailable — fail-open
        logger.warning(
            f"[MODEL RESOLUTION] Model catalog unavailable; using '{candidate}' "
            f"without endpoint validation for {endpoint_type}"
        )
        return candidate

    if candidate in model_api_support:
        supported = model_api_support[candidate]
        # Empty list = backward compat "supports all endpoints"
        if not supported or endpoint_type in supported:
            return candidate
        # Model exists but wrong endpoint — fall through to fallback

    # Step 5: Fallback
    logger.warning(
        f"[MODEL RESOLUTION] '{candidate}' does not support {endpoint_type} "
        f"(or is not in catalog); falling back to {FALLBACK_MODEL}. "
        f"Original incoming: '{incoming_model}'"
    )
    return FALLBACK_MODEL
