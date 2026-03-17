"""Tests for OCAChatModel model_api_support caching."""
import pytest
from unittest.mock import MagicMock, create_autospec, patch

from core.oauth2_token_manager import OCAOauth2TokenManager


def _make_token_manager():
    # Use create_autospec so isinstance(tm, OCAOauth2TokenManager) passes pydantic v2 validation
    tm = create_autospec(OCAOauth2TokenManager, instance=True)
    tm.get_access_token.return_value = "fake-token"
    return tm


CATALOG_RESPONSE = {
    "data": [
        {
            "litellm_params": {"model": "oca/gpt-4.1"},
            "model_info": {"supported_api_list": ["CHAT_COMPLETIONS", "RESPONSES"]},
        },
        {
            "litellm_params": {"model": "oca/gpt-5-codex"},
            "model_info": {"supported_api_list": ["responses"]},  # lowercase — must be uppercased
        },
        {
            "litellm_params": {"model": "oca/gpt-legacy"},
            "model_info": {},  # no supported_api_list key
        },
        {
            "litellm_params": {"model": "oca/gpt-5.4"},
            "model_info": {"supported_api_list": ["CHAT_COMPLETIONS", "RESPONSES"]},
        },
    ]
}


def test_fetch_populates_model_api_support():
    tm = _make_token_manager()
    mock_resp = MagicMock()
    mock_resp.json.return_value = CATALOG_RESPONSE
    mock_resp.raise_for_status.return_value = None
    tm.request.return_value = mock_resp

    from core.llm import OCAChatModel
    model = OCAChatModel(
        api_url="http://fake",
        model="oca/gpt-5.4",
        temperature=0.7,
        token_manager=tm,
        models_api_url="http://fake/models",
    )

    assert "oca/gpt-4.1" in model.model_api_support
    assert model.model_api_support["oca/gpt-4.1"] == ["CHAT_COMPLETIONS", "RESPONSES"]


def test_fetch_uppercases_endpoint_names():
    tm = _make_token_manager()
    mock_resp = MagicMock()
    mock_resp.json.return_value = CATALOG_RESPONSE
    mock_resp.raise_for_status.return_value = None
    tm.request.return_value = mock_resp

    from core.llm import OCAChatModel
    model = OCAChatModel(
        api_url="http://fake",
        model="oca/gpt-5.4",
        temperature=0.7,
        token_manager=tm,
        models_api_url="http://fake/models",
    )
    assert "RESPONSES" in model.model_api_support["oca/gpt-5-codex"]
    assert "responses" not in model.model_api_support["oca/gpt-5-codex"]


def test_fetch_missing_supported_api_list_stored_as_empty_list():
    tm = _make_token_manager()
    mock_resp = MagicMock()
    mock_resp.json.return_value = CATALOG_RESPONSE
    mock_resp.raise_for_status.return_value = None
    tm.request.return_value = mock_resp

    from core.llm import OCAChatModel
    model = OCAChatModel(
        api_url="http://fake",
        model="oca/gpt-5.4",
        temperature=0.7,
        token_manager=tm,
        models_api_url="http://fake/models",
    )
    assert model.model_api_support["oca/gpt-legacy"] == []


def test_repeated_fetch_does_not_duplicate_entries():
    tm = _make_token_manager()
    mock_resp = MagicMock()
    mock_resp.json.return_value = CATALOG_RESPONSE
    mock_resp.raise_for_status.return_value = None
    tm.request.return_value = mock_resp

    from core.llm import OCAChatModel
    model = OCAChatModel(
        api_url="http://fake",
        model="oca/gpt-5.4",
        temperature=0.7,
        token_manager=tm,
        models_api_url="http://fake/models",
    )
    initial_count = len(model.available_models)
    model.fetch_available_models()  # second call
    assert len(model.available_models) == initial_count


def test_fetch_failure_preserves_prior_cache():
    tm = _make_token_manager()
    mock_resp_ok = MagicMock()
    mock_resp_ok.json.return_value = CATALOG_RESPONSE
    mock_resp_ok.raise_for_status.return_value = None

    from core.llm import OCAChatModel
    # First call succeeds
    tm.request.return_value = mock_resp_ok
    model = OCAChatModel(
        api_url="http://fake",
        model="oca/gpt-5.4",
        temperature=0.7,
        token_manager=tm,
        models_api_url="http://fake/models",
    )
    assert len(model.available_models) > 0
    assert len(model.model_api_support) > 0

    # Second call fails
    tm.request.side_effect = ConnectionError("network down")
    prior_models = list(model.available_models)
    prior_support = dict(model.model_api_support)
    model.fetch_available_models()

    assert model.available_models == prior_models
    assert model.model_api_support == prior_support


def test_first_fetch_failure_model_api_support_stays_empty():
    """On first fetch failure, model_api_support stays {}, available_models falls back to [self.model]."""
    tm = _make_token_manager()
    tm.request.side_effect = ConnectionError("network down")

    from core.llm import OCAChatModel
    # Construct with models_api_url=None so fetch is skipped — gives us a valid instance
    model = OCAChatModel(
        api_url="http://fake",
        model="oca/gpt-5.4",
        temperature=0.7,
        token_manager=tm,
        models_api_url=None,
    )
    # Now simulate a first-call failure with models_api_url set
    model.models_api_url = "http://fake/models"
    model.available_models = []
    model.model_api_support = {}
    model.fetch_available_models()

    # model_api_support stays {} (no catalog data) — fail-open signal
    assert model.model_api_support == {}
    # available_models gets startup fallback to [self.model]
    assert model.available_models == ["oca/gpt-5.4"]


def test_empty_data_success_leaves_model_api_support_empty():
    """When fetch succeeds but returns empty data list, model_api_support is {} (fail-open)."""
    tm = _make_token_manager()
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"data": []}
    mock_resp.raise_for_status.return_value = None
    tm.request.return_value = mock_resp

    from core.llm import OCAChatModel
    model = OCAChatModel(
        api_url="http://fake",
        model="oca/gpt-5.4",
        temperature=0.7,
        token_manager=tm,
        models_api_url=None,
    )
    model.models_api_url = "http://fake/models"
    model.fetch_available_models()

    assert model.available_models == []
    assert model.model_api_support == {}


def test_fallback_model_absent_logs_warning():
    tm = _make_token_manager()
    mock_resp = MagicMock()
    # Catalog without oca/gpt-5.4
    mock_resp.json.return_value = {
        "data": [
            {
                "litellm_params": {"model": "oca/gpt-4.1"},
                "model_info": {"supported_api_list": ["CHAT_COMPLETIONS", "RESPONSES"]},
            }
        ]
    }
    mock_resp.raise_for_status.return_value = None
    tm.request.return_value = mock_resp

    from core.llm import OCAChatModel
    with patch("core.llm.logger") as mock_log:
        model = OCAChatModel(
            api_url="http://fake",
            model="oca/gpt-4.1",
            temperature=0.7,
            token_manager=tm,
            models_api_url="http://fake/models",
        )
    warning_calls = [str(c) for c in mock_log.warning.call_args_list]
    assert any("oca/gpt-5.4" in w and "not in the model catalog" in w.lower() for w in warning_calls)


def test_fallback_model_missing_chat_completions_logs_warning():
    tm = _make_token_manager()
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "data": [
            {
                "litellm_params": {"model": "oca/gpt-5.4"},
                "model_info": {"supported_api_list": ["RESPONSES"]},  # no CHAT_COMPLETIONS
            }
        ]
    }
    mock_resp.raise_for_status.return_value = None
    tm.request.return_value = mock_resp

    from core.llm import OCAChatModel
    with patch("core.llm.logger") as mock_log:
        model = OCAChatModel(
            api_url="http://fake",
            model="oca/gpt-5.4",
            temperature=0.7,
            token_manager=tm,
            models_api_url="http://fake/models",
        )
    warning_calls = [str(c) for c in mock_log.warning.call_args_list]
    assert any("CHAT_COMPLETIONS" in w for w in warning_calls)


def test_fallback_model_missing_responses_logs_warning():
    tm = _make_token_manager()
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "data": [
            {
                "litellm_params": {"model": "oca/gpt-5.4"},
                "model_info": {"supported_api_list": ["CHAT_COMPLETIONS"]},  # no RESPONSES
            }
        ]
    }
    mock_resp.raise_for_status.return_value = None
    tm.request.return_value = mock_resp

    from core.llm import OCAChatModel
    with patch("core.llm.logger") as mock_log:
        model = OCAChatModel(
            api_url="http://fake",
            model="oca/gpt-5.4",
            temperature=0.7,
            token_manager=tm,
            models_api_url="http://fake/models",
        )
    warning_calls = [str(c) for c in mock_log.warning.call_args_list]
    assert any("RESPONSES" in w for w in warning_calls)


def test_fallback_model_empty_supported_list_no_warning():
    """Empty supported_api_list on FALLBACK_MODEL means 'supports all' — no warning."""
    tm = _make_token_manager()
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "data": [
            {
                "litellm_params": {"model": "oca/gpt-5.4"},
                "model_info": {},  # no supported_api_list = empty list = supports all
            }
        ]
    }
    mock_resp.raise_for_status.return_value = None
    tm.request.return_value = mock_resp

    from core.llm import OCAChatModel
    with patch("core.llm.logger") as mock_log:
        model = OCAChatModel(
            api_url="http://fake",
            model="oca/gpt-5.4",
            temperature=0.7,
            token_manager=tm,
            models_api_url="http://fake/models",
        )
    warning_calls = [str(c) for c in mock_log.warning.call_args_list]
    assert not any("CHAT_COMPLETIONS" in w or "RESPONSES" in w for w in warning_calls)
