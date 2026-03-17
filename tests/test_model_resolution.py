"""
Tests for resolve_model_for_endpoint (replaces resolve_model_name tests).
"""
from unittest.mock import patch
import pytest
from model_resolver import resolve_model_for_endpoint, FALLBACK_MODEL


CATALOG = {
    "oca/gpt-4.1": ["CHAT_COMPLETIONS", "RESPONSES"],
    "oca/gpt-5-codex": ["RESPONSES"],
    "oca/gpt-oss-120b": ["CHAT_COMPLETIONS"],
    "oca/gpt-5.4": ["CHAT_COMPLETIONS", "RESPONSES"],
    "oca/gpt-legacy": [],  # empty list = supports all endpoints (backward compat)
}


class TestModelResolutionNoPrefixNormalization:
    def test_incoming_with_oca_prefix_supported(self):
        with patch("model_resolver._get_runtime_env_value", return_value=""):
            assert resolve_model_for_endpoint("oca/gpt-4.1", "LLM_MODEL_NAME", "CHAT_COMPLETIONS", CATALOG) == "oca/gpt-4.1"

    def test_incoming_without_oca_prefix_adds_prefix_if_supported(self):
        with patch("model_resolver._get_runtime_env_value", return_value=""):
            assert resolve_model_for_endpoint("gpt-4.1", "LLM_MODEL_NAME", "CHAT_COMPLETIONS", CATALOG) == "oca/gpt-4.1"

    def test_incoming_with_whitespace_stripped(self):
        with patch("model_resolver._get_runtime_env_value", return_value=""):
            assert resolve_model_for_endpoint("  gpt-4.1  ", "LLM_MODEL_NAME", "CHAT_COMPLETIONS", CATALOG) == "oca/gpt-4.1"

    def test_incoming_double_oca_prefix_normalized(self):
        with patch("model_resolver._get_runtime_env_value", return_value=""):
            assert resolve_model_for_endpoint("oca/oca/gpt-4.1", "LLM_MODEL_NAME", "CHAT_COMPLETIONS", CATALOG) == "oca/gpt-4.1"


class TestModelResolutionFallback:
    def test_none_incoming_falls_back(self):
        with patch("model_resolver._get_runtime_env_value", return_value=""):
            assert resolve_model_for_endpoint(None, "LLM_MODEL_NAME", "CHAT_COMPLETIONS", CATALOG) == FALLBACK_MODEL

    def test_empty_incoming_falls_back(self):
        with patch("model_resolver._get_runtime_env_value", return_value=""):
            assert resolve_model_for_endpoint("", "LLM_MODEL_NAME", "CHAT_COMPLETIONS", CATALOG) == FALLBACK_MODEL

    def test_unsupported_endpoint_falls_back(self):
        # gpt-5-codex is RESPONSES-only; using for CHAT_COMPLETIONS → fallback
        with patch("model_resolver._get_runtime_env_value", return_value=""):
            assert resolve_model_for_endpoint("gpt-5-codex", "LLM_MODEL_NAME", "CHAT_COMPLETIONS", CATALOG) == FALLBACK_MODEL

    def test_unknown_model_falls_back(self):
        with patch("model_resolver._get_runtime_env_value", return_value=""):
            assert resolve_model_for_endpoint("gpt-unknown", "LLM_MODEL_NAME", "CHAT_COMPLETIONS", CATALOG) == FALLBACK_MODEL

    def test_empty_supported_api_list_treated_as_supports_all(self):
        # oca/gpt-legacy has supported_api_list=[] — must be usable for any endpoint (backward compat)
        with patch("model_resolver._get_runtime_env_value", return_value=""):
            assert resolve_model_for_endpoint("gpt-legacy", "LLM_MODEL_NAME", "CHAT_COMPLETIONS", CATALOG) == "oca/gpt-legacy"

    def test_empty_supported_api_list_supports_responses_too(self):
        with patch("model_resolver._get_runtime_env_value", return_value=""):
            assert resolve_model_for_endpoint("gpt-legacy", "LLM_RESPONSES_MODEL_NAME", "RESPONSES", CATALOG) == "oca/gpt-legacy"


class TestModelResolutionEnvOverride:
    def test_env_override_used_when_set(self):
        with patch("model_resolver._get_runtime_env_value", return_value="oca/gpt-5.4"):
            assert resolve_model_for_endpoint("gpt-5-codex", "LLM_MODEL_NAME", "CHAT_COMPLETIONS", CATALOG) == "oca/gpt-5.4"

    def test_env_override_normalized_without_prefix(self):
        with patch("model_resolver._get_runtime_env_value", return_value="gpt-4.1"):
            assert resolve_model_for_endpoint("gpt-5-codex", "LLM_MODEL_NAME", "CHAT_COMPLETIONS", CATALOG) == "oca/gpt-4.1"

    def test_env_override_wrong_endpoint_raises_500_error(self):
        with patch("model_resolver._get_runtime_env_value", return_value="oca/gpt-5-codex"):
            with pytest.raises(ValueError) as exc_info:
                resolve_model_for_endpoint("gpt-4.1", "LLM_MODEL_NAME", "CHAT_COMPLETIONS", CATALOG)
        assert "CHAT_COMPLETIONS" in str(exc_info.value)

    def test_env_override_not_in_catalog_raises_error(self):
        with patch("model_resolver._get_runtime_env_value", return_value="oca/nonexistent"):
            with pytest.raises(ValueError) as exc_info:
                resolve_model_for_endpoint("gpt-4.1", "LLM_MODEL_NAME", "CHAT_COMPLETIONS", CATALOG)
        assert "not in the model catalog" in str(exc_info.value)


class TestModelResolutionEmptyCatalog:
    def test_empty_catalog_uses_prefixed_incoming_fail_open(self):
        with patch("model_resolver._get_runtime_env_value", return_value=""):
            assert resolve_model_for_endpoint("gpt-4.1", "LLM_MODEL_NAME", "CHAT_COMPLETIONS", {}) == "oca/gpt-4.1"

    def test_empty_catalog_env_override_used_without_validation(self):
        with patch("model_resolver._get_runtime_env_value", return_value="oca/gpt-4.1"):
            assert resolve_model_for_endpoint("gpt-5-codex", "LLM_MODEL_NAME", "CHAT_COMPLETIONS", {}) == "oca/gpt-4.1"
