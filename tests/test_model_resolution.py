"""
Tests for Model Name Resolution

This module tests the model name resolution logic in responses_api.py.
"""

from unittest.mock import patch

from responses_api import resolve_model_name


class TestModelResolution:
    """Test model name resolution logic."""

    def test_incoming_model_with_oca_prefix(self):
        """Test that models with oca/ prefix are used as-is."""
        result = resolve_model_name("oca/gpt-4o")
        assert result == "oca/gpt-4o"

    def test_incoming_model_with_oca_prefix_whitespace(self):
        """Test that whitespace is stripped from oca/ models."""
        result = resolve_model_name("  oca/gpt-4o  ")
        assert result == "oca/gpt-4o"

    def test_incoming_model_with_oca_prefix_case_insensitive(self):
        """Test that OCA/ prefix works (case insensitive)."""
        result = resolve_model_name("OCA/gpt-4o")
        assert result == "OCA/gpt-4o"

    def test_incoming_model_without_prefix_uses_default(self):
        """Test that models without oca/ prefix fall back to default."""
        # Assuming LLM_MODEL_NAME is set in test environment
        result = resolve_model_name("gpt-4o")
        # Should return the default model
        assert "oca/" in result

    def test_incoming_model_empty_string(self):
        """Test that empty string falls back to default."""
        result = resolve_model_name("")
        # Should return the default model
        assert "oca/" in result

    def test_incoming_model_none_ignored(self):
        """Test that None is handled gracefully."""
        result = resolve_model_name(None)
        # Should return None or default
        assert result is None or "oca/" in result


class TestModelResolutionWithEnv:
    """Test model resolution with different env configurations."""

    @patch("responses_api._get_default_model", return_value="oca/gpt-5.2")
    def test_fallback_to_specific_default(self, _mock_get_default_model):
        """Test fallback to a specific default model."""
        result = resolve_model_name("gpt-4")
        assert result == "oca/gpt-5.2"

    @patch("responses_api._get_default_model", return_value="")
    def test_no_default_set(self, _mock_get_default_model):
        """Test behavior when no default is set."""
        result = resolve_model_name("gpt-4o")
        # Should return incoming as-is
        assert result == "gpt-4o"

    @patch("responses_api._get_default_model", return_value="oca/custom-model")
    def test_fallback_to_custom_model(self, _mock_get_default_model):
        """Test fallback to custom default model."""
        result = resolve_model_name("some-random-model")
        assert result == "oca/custom-model"


class TestModelResolutionEdgeCases:
    """Test edge cases for model resolution."""

    def test_various_oca_prefix_formats(self):
        """Test various oca/ prefix formats."""
        # Standard format
        assert resolve_model_name("oca/gpt-4o") == "oca/gpt-4o"
        # With whitespace
        assert resolve_model_name("  oca/gpt-4o  ") == "oca/gpt-4o"
        # Mixed case
        result = resolve_model_name("Oca/GPT-4o")
        assert result.startswith("oca/") or result.startswith("Oca/")

    def test_model_with_version_numbers(self):
        """Test models with version numbers."""
        # These should fall back to default (no oca/ prefix)
        with patch("responses_api._get_default_model", return_value="oca/gpt-5.2"):
            assert resolve_model_name("gpt-5.2") == "oca/gpt-5.2"
            assert resolve_model_name("gpt-5.1-codex") == "oca/gpt-5.2"
            assert resolve_model_name("llama4") == "oca/gpt-5.2"

    def test_model_with_different_prefixes(self):
        """Test models with different prefixes."""
        with patch("responses_api._get_default_model", return_value="oca/gpt-4o"):
            # These should all fall back to default
            assert resolve_model_name("openai/gpt-4") == "oca/gpt-4o"
            assert resolve_model_name("anthropic/claude") == "oca/gpt-4o"
            assert resolve_model_name("google/gemini") == "oca/gpt-4o"


class TestModelResolutionIntegration:
    """Integration tests for model resolution."""

    def test_resolution_logs_info(self):
        """Test that resolution logs appropriate messages."""
        with patch("responses_api._get_default_model", return_value="oca/gpt-5.2"), \
             patch("responses_api.logger.info") as mock_info:
            result = resolve_model_name("gpt-4o")

        assert result == "oca/gpt-5.2"
        assert mock_info.call_count == 1
        log_message = mock_info.call_args[0][0]
        assert "MODEL RESOLUTION" in log_message
        assert "gpt-4o" in log_message
        assert "oca/gpt-5.2" in log_message

    def test_no_log_when_using_oca_prefix(self):
        """Test that no resolution log when model already has oca/ prefix."""
        with patch("responses_api.logger.info") as mock_info:
            result = resolve_model_name("oca/gpt-4o")

        assert result == "oca/gpt-4o"
        assert mock_info.call_count == 0
