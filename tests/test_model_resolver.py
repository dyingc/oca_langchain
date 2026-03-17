import pytest
from unittest.mock import patch
from model_resolver import resolve_model_for_endpoint, FALLBACK_MODEL

CATALOG = {
    "oca/gpt-4.1": ["CHAT_COMPLETIONS", "RESPONSES"],
    "oca/gpt-5-codex": ["RESPONSES"],
    "oca/gpt-oss-120b": ["CHAT_COMPLETIONS"],
    "oca/gpt-5.4": ["CHAT_COMPLETIONS", "RESPONSES"],
    "oca/gpt-legacy": [],  # empty list = supports all endpoints (backward compat)
}

# ── Fallback (no catalog, no env) ─────────────────────────────────────────────

def test_none_incoming_returns_fallback():
    assert resolve_model_for_endpoint(None, "LLM_MODEL_NAME", "CHAT_COMPLETIONS", CATALOG) == FALLBACK_MODEL


def test_empty_incoming_returns_fallback():
    assert resolve_model_for_endpoint("", "LLM_MODEL_NAME", "CHAT_COMPLETIONS", CATALOG) == FALLBACK_MODEL


def test_whitespace_incoming_returns_fallback():
    assert resolve_model_for_endpoint("   ", "LLM_MODEL_NAME", "CHAT_COMPLETIONS", CATALOG) == FALLBACK_MODEL


# ── Prefix normalisation ──────────────────────────────────────────────────────

def test_adds_oca_prefix_if_missing_and_supported():
    with patch("model_resolver._get_runtime_env_value", return_value=""):
        result = resolve_model_for_endpoint("gpt-4.1", "LLM_MODEL_NAME", "CHAT_COMPLETIONS", CATALOG)
    assert result == "oca/gpt-4.1"


def test_incoming_already_prefixed_and_supported():
    with patch("model_resolver._get_runtime_env_value", return_value=""):
        result = resolve_model_for_endpoint("oca/gpt-4.1", "LLM_MODEL_NAME", "CHAT_COMPLETIONS", CATALOG)
    assert result == "oca/gpt-4.1"


def test_double_prefix_normalised():
    with patch("model_resolver._get_runtime_env_value", return_value=""):
        result = resolve_model_for_endpoint("oca/oca/gpt-4.1", "LLM_MODEL_NAME", "CHAT_COMPLETIONS", CATALOG)
    assert result == "oca/gpt-4.1"


# ── Endpoint support check ────────────────────────────────────────────────────

def test_model_unsupported_for_endpoint_falls_back():
    # gpt-5-codex only supports RESPONSES, not CHAT_COMPLETIONS
    with patch("model_resolver._get_runtime_env_value", return_value=""):
        result = resolve_model_for_endpoint("gpt-5-codex", "LLM_MODEL_NAME", "CHAT_COMPLETIONS", CATALOG)
    assert result == FALLBACK_MODEL


def test_model_supported_for_responses():
    with patch("model_resolver._get_runtime_env_value", return_value=""):
        result = resolve_model_for_endpoint("gpt-5-codex", "LLM_RESPONSES_MODEL_NAME", "RESPONSES", CATALOG)
    assert result == "oca/gpt-5-codex"


def test_model_not_in_catalog_falls_back():
    with patch("model_resolver._get_runtime_env_value", return_value=""):
        result = resolve_model_for_endpoint("gpt-unknown", "LLM_MODEL_NAME", "CHAT_COMPLETIONS", CATALOG)
    assert result == FALLBACK_MODEL


# ── Backward compat: empty supported_api_list ─────────────────────────────────

def test_empty_supported_api_list_means_supports_all():
    with patch("model_resolver._get_runtime_env_value", return_value=""):
        result = resolve_model_for_endpoint("gpt-legacy", "LLM_MODEL_NAME", "CHAT_COMPLETIONS", CATALOG)
    assert result == "oca/gpt-legacy"


def test_empty_supported_api_list_means_supports_responses_too():
    with patch("model_resolver._get_runtime_env_value", return_value=""):
        result = resolve_model_for_endpoint("gpt-legacy", "LLM_RESPONSES_MODEL_NAME", "RESPONSES", CATALOG)
    assert result == "oca/gpt-legacy"


# ── Empty catalog (fail-open) ──────────────────────────────────────────────────

def test_empty_catalog_fail_open_uses_prefixed_incoming():
    with patch("model_resolver._get_runtime_env_value", return_value=""):
        result = resolve_model_for_endpoint("gpt-4.1", "LLM_MODEL_NAME", "CHAT_COMPLETIONS", {})
    assert result == "oca/gpt-4.1"


def test_empty_catalog_fail_open_logs_warning():
    with patch("model_resolver._get_runtime_env_value", return_value=""), \
         patch("model_resolver.logger.warning") as mock_warn:
        resolve_model_for_endpoint("gpt-4.1", "LLM_MODEL_NAME", "CHAT_COMPLETIONS", {})
    assert mock_warn.call_count >= 1
    assert any("catalog unavailable" in str(c).lower() or "without endpoint" in str(c).lower()
               for c in mock_warn.call_args_list)


# ── Env override — happy path ─────────────────────────────────────────────────

def test_env_override_used_when_set_and_supported():
    with patch("model_resolver._get_runtime_env_value", return_value="oca/gpt-4.1"):
        result = resolve_model_for_endpoint("gpt-5-codex", "LLM_MODEL_NAME", "CHAT_COMPLETIONS", CATALOG)
    assert result == "oca/gpt-4.1"


def test_env_override_normalized_with_oca_prefix():
    # Operator sets LLM_MODEL_NAME=gpt-4.1 without oca/ prefix
    with patch("model_resolver._get_runtime_env_value", return_value="gpt-4.1"):
        result = resolve_model_for_endpoint("gpt-5-codex", "LLM_MODEL_NAME", "CHAT_COMPLETIONS", CATALOG)
    assert result == "oca/gpt-4.1"


# ── Env override — misconfiguration (raises ValueError) ──────────────────────

def test_env_override_not_in_catalog_raises_value_error():
    with patch("model_resolver._get_runtime_env_value", return_value="oca/gpt-unknown"):
        with pytest.raises(ValueError) as exc_info:
            resolve_model_for_endpoint("gpt-4.1", "LLM_MODEL_NAME", "CHAT_COMPLETIONS", CATALOG)
    assert "not in the model catalog" in str(exc_info.value)
    assert "LLM_MODEL_NAME" in str(exc_info.value)


def test_env_override_wrong_endpoint_raises_value_error():
    # gpt-5-codex only supports RESPONSES, so using it as LLM_MODEL_NAME (CHAT_COMPLETIONS) is wrong
    with patch("model_resolver._get_runtime_env_value", return_value="oca/gpt-5-codex"):
        with pytest.raises(ValueError) as exc_info:
            resolve_model_for_endpoint("gpt-4.1", "LLM_MODEL_NAME", "CHAT_COMPLETIONS", CATALOG)
    assert "CHAT_COMPLETIONS" in str(exc_info.value)
    assert "RESPONSES" in str(exc_info.value)  # supported endpoint listed in error


def test_env_override_empty_catalog_warns_and_uses_override():
    # Catalog unavailable: cannot validate, fail-open
    with patch("model_resolver._get_runtime_env_value", return_value="oca/gpt-4.1"), \
         patch("model_resolver.logger.warning") as mock_warn:
        result = resolve_model_for_endpoint("gpt-5-codex", "LLM_MODEL_NAME", "CHAT_COMPLETIONS", {})
    assert result == "oca/gpt-4.1"
    assert mock_warn.call_count >= 1
