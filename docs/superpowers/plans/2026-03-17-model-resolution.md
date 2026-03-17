# Model Resolution Logic Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add endpoint-aware model resolution to `/v1/chat/completions` and `/v1/responses` so clients that send model names without the `oca/` prefix (e.g., Codex CLI) are automatically matched to a supported catalog model instead of receiving 404 errors.

**Architecture:** Two new root-level modules — `runtime_env.py` (env reading, centralises the two duplicate `_get_runtime_env_value` implementations) and `model_resolver.py` (pure resolution function accepting model catalog as parameter). Existing `OCAChatModel` is extended to cache `model_api_support: Dict[str, List[str]]` at startup. All three API handler files call `resolve_model_for_endpoint()` from `model_resolver.py`.

**Tech Stack:** Python 3.13, FastAPI, LangChain BaseChatModel (Pydantic v1 compat), pytest, python-dotenv

---

## Chunk 1: New modules — `runtime_env.py` and `model_resolver.py`

### Task 1: Create `runtime_env.py`

**Files:**
- Create: `runtime_env.py`
- Test: `tests/test_runtime_env.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_runtime_env.py
import os
import pytest
from runtime_env import _get_runtime_env_value


def test_reads_value_from_env_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text('MY_KEY="hello"\n', encoding="utf-8")
    monkeypatch.setattr("runtime_env._ENV_PATH", str(env_file))
    monkeypatch.delenv("MY_KEY", raising=False)
    assert _get_runtime_env_value("MY_KEY") == "hello"


def test_returns_default_when_key_absent_from_env_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("# no key\n", encoding="utf-8")
    monkeypatch.setattr("runtime_env._ENV_PATH", str(env_file))
    monkeypatch.delenv("MY_KEY", raising=False)
    assert _get_runtime_env_value("MY_KEY", "fallback") == "fallback"


def test_falls_back_to_os_environ_when_file_missing(monkeypatch):
    monkeypatch.setattr("runtime_env._ENV_PATH", "/nonexistent/.env")
    monkeypatch.setenv("MY_KEY", "from_os")
    assert _get_runtime_env_value("MY_KEY") == "from_os"


def test_strips_whitespace_from_value(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text('MY_KEY="  spaced  "\n', encoding="utf-8")
    monkeypatch.setattr("runtime_env._ENV_PATH", str(env_file))
    monkeypatch.delenv("MY_KEY", raising=False)
    assert _get_runtime_env_value("MY_KEY") == "spaced"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/yingdong/VSCode/oca_langchain && source .venv/bin/activate
pytest tests/test_runtime_env.py -v
```

Expected: `ModuleNotFoundError: No module named 'runtime_env'`

- [ ] **Step 3: Create `runtime_env.py`**

```python
import os
from dotenv import dotenv_values

_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")


def _get_runtime_env_value(key: str, default: str = "") -> str:
    """Read env value with .env as source of truth when the file exists.

    This avoids stale os.environ values when a key is removed from .env
    while the process is still running.
    """
    if os.path.exists(_ENV_PATH):
        values = dotenv_values(_ENV_PATH)
        value = values.get(key)
        if value is None:
            return default
        return str(value).strip()
    return os.getenv(key, default).strip()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_runtime_env.py -v
```

Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add runtime_env.py tests/test_runtime_env.py
git commit -m "feat: add runtime_env module centralising _get_runtime_env_value"
```

---

### Task 2: Create `model_resolver.py` — core resolution logic

**Files:**
- Create: `model_resolver.py`
- Test: `tests/test_model_resolver.py`

- [ ] **Step 1: Write all failing tests**

```python
# tests/test_model_resolver.py
import pytest
from unittest.mock import patch
from model_resolver import resolve_model_for_endpoint, FALLBACK_MODEL

CATALOG = {
    "oca/gpt-4.1": ["CHAT_COMPLETIONS", "RESPONSES"],
    "oca/gpt-5-codex": ["RESPONSES"],
    "oca/gpt-oss-120b": ["CHAT_COMPLETIONS"],
    "oca/gpt-5.4": ["CHAT_COMPLETIONS", "RESPONSES"],
    "oca/gpt-legacy": [],  # empty list = supports all (backward compat)
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_model_resolver.py -v
```

Expected: `ModuleNotFoundError: No module named 'model_resolver'`

- [ ] **Step 3: Create `model_resolver.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_model_resolver.py -v
```

Expected: all tests pass

- [ ] **Step 5: Commit**

```bash
git add model_resolver.py tests/test_model_resolver.py
git commit -m "feat: add model_resolver with endpoint-aware resolution logic"
```

---

## Chunk 2: Update `core/llm.py` — OCAChatModel

### Task 3: Add `model_api_support` field and atomic update in `fetch_available_models()`

**Files:**
- Modify: `core/llm.py:344-438`
- Test: `tests/test_llm_model_api_support.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_llm_model_api_support.py
"""Tests for OCAChatModel model_api_support caching."""
import pytest
from unittest.mock import MagicMock, patch


def _make_model(token_manager):
    """Create OCAChatModel with mocked dependencies."""
    from core.llm import OCAChatModel
    return OCAChatModel(
        api_url="http://fake",
        model="oca/gpt-5.4",
        temperature=0.7,
        token_manager=token_manager,
        models_api_url=None,  # skip fetch in __init__
    )


def _make_token_manager():
    tm = MagicMock()
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
    ]
}


def test_fetch_populates_model_api_support(tmp_path, monkeypatch):
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
    # "responses" in catalog should be stored as "RESPONSES"
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
    import requests as req_lib
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
    # Construct with models_api_url=None so fetch is skipped — gives us a valid instance to test with
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

    # model_api_support stays {} (no catalog data) — this is the "fail-open" signal
    assert model.model_api_support == {}
    # available_models gets startup fallback to [self.model] so __init__ validation passes
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


def test_fallback_model_missing_chat_completions_logs_warning():
    """Warn if FALLBACK_MODEL is in catalog but doesn't support CHAT_COMPLETIONS."""
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
    """Warn if FALLBACK_MODEL is in catalog but doesn't support RESPONSES."""
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
    # No warnings about CHAT_COMPLETIONS or RESPONSES should be emitted
    warning_calls = [str(c) for c in mock_log.warning.call_args_list]
    assert not any("CHAT_COMPLETIONS" in w or "RESPONSES" in w for w in warning_calls)


def test_fallback_model_absent_logs_warning():
    tm = _make_token_manager()
    mock_resp = MagicMock()
    # Catalog without oca/gpt-5.4 (the FALLBACK_MODEL)
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_llm_model_api_support.py -v
```

Expected: failures on `model_api_support` attribute not existing

- [ ] **Step 3: Update `core/llm.py`**

Three changes required:

**3a. Add import at top of file** (after existing imports):
```python
from typing import Any, Dict, List, Mapping, Optional, Iterator, AsyncIterator
```
(replace existing `from typing import Any, List, ...` — add `Dict`)

**3b. Update `OCAChatModel` field declarations** (lines 356-357):
```python
# Before:
available_models: List[str] = []

# After:
from langchain_core.pydantic_v1 import Field
available_models: List[str] = Field(default_factory=list)
model_api_support: Dict[str, List[str]] = Field(default_factory=dict)
```

**3c. Replace `fetch_available_models()` body** (lines 377-438) with atomic-update version:

```python
def fetch_available_models(self):
    """Fetch and cache the available models list and their supported endpoint types."""
    if not self.models_api_url:
        if self._debug:
            print("Warning: LLM_MODELS_API_URL is not configured, cannot fetch models dynamically.")
        if self.model:
            self.available_models = [self.model]
        return

    headers = {
        "Authorization": f"Bearer {self.token_manager.get_access_token()}",
        "Accept": "application/json",
    }

    max_retries = 3
    retry_delay = 3

    for attempt in range(max_retries):
        try:
            if self._debug:
                print(f"Fetching available models from {self.models_api_url} (attempt {attempt + 1}/{max_retries})...")

            response = self.token_manager.request(
                method="GET",
                url=self.models_api_url,
                headers=headers,
                _do_retry=True
            )
            response.raise_for_status()

            models_data = response.json().get("data", [])

            # Use temp snapshots — only commit to self.* on full success
            # This preserves prior cache on partial failure and prevents double-prefix accumulation
            new_available = []
            new_api_support = {}

            for model in models_data:
                model_id = model.get("id") or model.get("litellm_params", {}).get("model")
                if model_id:
                    new_available.append(model_id)
                    supported = model.get("model_info", {}).get("supported_api_list") or []
                    new_api_support[model_id] = [s.upper() for s in supported]

            # Atomic commit: both fields replaced together only on success
            self.available_models = new_available
            self.model_api_support = new_api_support

            if not self.available_models:
                if self._debug:
                    print("Warning: The API returned an empty models list.")
            else:
                if self._debug:
                    print(f"Successfully retrieved {len(self.available_models)} available models.")

                # Startup validation: warn if fallback model has issues
                from model_resolver import FALLBACK_MODEL
                fallback_support = new_api_support.get(FALLBACK_MODEL)
                if fallback_support is None:
                    logger.warning(
                        f"Fallback model '{FALLBACK_MODEL}' is not in the model catalog. "
                        f"Requests routed to it will fail downstream."
                    )
                else:
                    if fallback_support and "CHAT_COMPLETIONS" not in fallback_support:
                        logger.warning(
                            f"Fallback model '{FALLBACK_MODEL}' does not support CHAT_COMPLETIONS."
                        )
                    if fallback_support and "RESPONSES" not in fallback_support:
                        logger.warning(
                            f"Fallback model '{FALLBACK_MODEL}' does not support RESPONSES."
                        )

            return  # Success

        except (ConnectionError, httpx.ConnectError, httpx.ReadTimeout) as e:
            if attempt < max_retries - 1:
                if self._debug:
                    print(f"Failed to connect to models API. Retrying in {retry_delay} seconds... Reason: {e}")
                time.sleep(retry_delay)
            else:
                if self._debug:
                    print(f"Error: Unable to connect to models API after multiple attempts. Reason: {e}")
                # Do NOT update self.available_models or self.model_api_support
                # Prior cached values are preserved; if first call, they remain [] / {}
                # Exception: if first call failed, fall back to env model for startup validation
                if not self.available_models and self.model:
                    self.available_models = [self.model]

        except json.JSONDecodeError:
            if self._debug:
                print("Error: Failed to parse models API response; not a valid JSON format.")
            # Do NOT clear self.available_models — preserve prior cache
            # Exception: if this is the first call, set startup fallback so __init__ validation passes
            if not self.available_models and self.model:
                self.available_models = [self.model]
            return
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_llm_model_api_support.py -v
```

Expected: all tests pass

- [ ] **Step 5: Run full test suite to check no regressions**

```bash
pytest tests/ -v --ignore=tests/test_model_resolution.py --ignore=tests/test_dynamic_env_reload.py
```

Expected: all non-ignored tests pass (the two ignored files will be rewritten in Task 6)

- [ ] **Step 6: Commit**

```bash
git add core/llm.py tests/test_llm_model_api_support.py
git commit -m "feat: add model_api_support field and atomic fetch to OCAChatModel"
```

---

### Task 4: Normalize `LLM_MODEL_NAME` with `oca/` prefix in `from_env()`

**Files:**
- Modify: `core/llm.py:440-455`

- [ ] **Step 1: Write the failing test** (add to `tests/test_llm_model_api_support.py`):

```python
def test_from_env_normalizes_model_name_without_prefix(monkeypatch):
    """LLM_MODEL_NAME=gpt-4.1 should be normalized to oca/gpt-4.1 before startup validation."""
    monkeypatch.setenv("LLM_API_URL", "http://fake")
    monkeypatch.setenv("LLM_MODEL_NAME", "gpt-4.1")  # no oca/ prefix
    monkeypatch.setenv("LLM_MODELS_API_URL", "http://fake/models")

    tm = _make_token_manager()
    mock_resp = MagicMock()
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
    # Should NOT raise ValueError about model not in available_models
    model = OCAChatModel.from_env(tm)
    assert model.model == "oca/gpt-4.1"


def test_from_env_model_already_prefixed_unchanged(monkeypatch):
    """LLM_MODEL_NAME=oca/gpt-4.1 should stay as-is."""
    monkeypatch.setenv("LLM_API_URL", "http://fake")
    monkeypatch.setenv("LLM_MODEL_NAME", "oca/gpt-4.1")
    monkeypatch.setenv("LLM_MODELS_API_URL", "http://fake/models")

    tm = _make_token_manager()
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "data": [{"litellm_params": {"model": "oca/gpt-4.1"}, "model_info": {}}]
    }
    mock_resp.raise_for_status.return_value = None
    tm.request.return_value = mock_resp

    from core.llm import OCAChatModel
    model = OCAChatModel.from_env(tm)
    assert model.model == "oca/gpt-4.1"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_llm_model_api_support.py::test_from_env_normalizes_model_name_without_prefix -v
```

Expected: `ValueError: Error: The specified model 'gpt-4.1' is not in the list of available models`

- [ ] **Step 3: Update `from_env()` in `core/llm.py`** (lines 440-455):

```python
@classmethod
def from_env(cls, token_manager: OCAOauth2TokenManager, debug: bool = False) -> OCAChatModel:
    api_url = os.getenv("LLM_API_URL")
    model = os.getenv("LLM_MODEL_NAME", "").strip()
    # Normalize: add oca/ prefix if missing (same rule as model_resolver)
    if model and not model.lower().startswith("oca/"):
        model = f"oca/{model}"
    temperature = float(os.getenv("LLM_TEMPERATURE", 0.7))
    models_api_url = os.getenv("LLM_MODELS_API_URL")
    llm_request_timeout = float(os.getenv("LLM_REQUEST_TIMEOUT", 120.0))

    if not api_url:
        raise ValueError("Error: Please ensure .env contains LLM_API_URL.")
    if not models_api_url and not model:
        raise ValueError("Error: Either LLM_MODELS_API_URL must be set or LLM_MODEL_NAME must be provided in .env.")

    return cls(
        api_url=api_url, model=model, temperature=temperature,
        token_manager=token_manager, models_api_url=models_api_url,
        llm_request_timeout=llm_request_timeout, _debug=debug
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_llm_model_api_support.py -v
```

Expected: all tests pass

- [ ] **Step 5: Commit**

```bash
git add core/llm.py tests/test_llm_model_api_support.py
git commit -m "feat: normalize LLM_MODEL_NAME with oca/ prefix in OCAChatModel.from_env()"
```

---

## Chunk 3: Update API endpoints

### Task 5: Update `api.py` — replace 404 guard with model resolution

**Files:**
- Modify: `api.py:386-401`

- [ ] **Step 1: Verify current behavior (manual check)**

```bash
# Confirm line numbers match expectation
grep -n "not in chat_model.available_models\|chat_model.model = request.model" /Users/yingdong/VSCode/oca_langchain/api.py
```

Expected: lines near 394-401

- [ ] **Step 2: Add import and replace resolution logic**

In `api.py`, at the top imports section, add:
```python
from model_resolver import resolve_model_for_endpoint
```

Replace the block in `create_chat_completion()` (lines 393-401):
```python
# Before:
if request.model not in chat_model.available_models:
    raise HTTPException(
        status_code=404,
        detail=f"Model '{request.model}' not found. Available models: {', '.join(chat_model.available_models)}"
    )
chat_model.model = request.model

# After:
try:
    resolved_model = resolve_model_for_endpoint(
        request.model,
        "LLM_MODEL_NAME",
        "CHAT_COMPLETIONS",
        chat_model.model_api_support,
    )
except ValueError as e:
    raise HTTPException(status_code=500, detail=str(e))

chat_model.model = resolved_model
request.model = resolved_model
```

- [ ] **Step 3: Run existing tests to verify no regression**

```bash
pytest tests/ -v --ignore=tests/test_model_resolution.py --ignore=tests/test_dynamic_env_reload.py
```

Expected: all non-ignored tests pass

- [ ] **Step 4: Commit**

```bash
git add api.py
git commit -m "feat: replace 404 model guard with resolve_model_for_endpoint in chat completions"
```

---

### Task 6: Update `responses_api.py` — remove local resolver, use shared modules

**Files:**
- Modify: `responses_api.py:1-104`

- [ ] **Step 1: Identify all call sites**

```bash
grep -n "_get_runtime_env_value\|resolve_model_name\|LLM_MODEL_NAME" /Users/yingdong/VSCode/oca_langchain/responses_api.py
```

Confirm:
- `_get_runtime_env_value` defined locally at line 57, used at line 65
- `resolve_model_name()` defined at line 78, called in `create_response()`
- `LLM_MODEL_NAME` used in `_get_default_model()` at line 74

- [ ] **Step 2: Update `responses_api.py`**

Replace the "Model Resolution" section (lines 53-103):

**Remove:**
```python
# --- Model Resolution ---
_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")

def _get_runtime_env_value(key: str, default: str = "") -> str: ...
def _get_default_model() -> str: ...
def resolve_model_name(incoming_model: str) -> str: ...
```

**Add at top of file** (after existing imports):
```python
from runtime_env import _get_runtime_env_value
from model_resolver import resolve_model_for_endpoint
```

Remove the `dotenv` import if no longer used:
```python
# Remove: from dotenv import dotenv_values
```

Find the call site in `create_response()` (search for `resolve_model_name`).

**IMPORTANT**: Run `grep -n "_get_chat_model\|resolve_model_name" responses_api.py` first to confirm whether `chat_model = _get_chat_model()` is already assigned earlier in `create_response()`. In the current code it IS called at line ~501 before `resolve_model_name` at line ~505 — reuse that existing `chat_model` variable; do NOT call `_get_chat_model()` again.

Also remove the 404 guard block that follows `resolve_model_name` (lines ~507-518: `if request.model not in chat_model.available_models: raise HTTPException(404, ...)`). That guard is replaced by resolution logic.

```python
# Before (two statements to replace/remove):
request.model = resolve_model_name(request.model)
if request.model not in chat_model.available_models:
    raise HTTPException(
        status_code=404,
        detail=f"Model '{request.model}' not found. ..."
    )

# After (using already-assigned chat_model from earlier in the function):
try:
    request.model = resolve_model_for_endpoint(
        request.model,
        "LLM_RESPONSES_MODEL_NAME",
        "RESPONSES",
        chat_model.model_api_support,
    )
except ValueError as e:
    raise HTTPException(
        status_code=500,
        detail={"type": "error", "error": {"type": "configuration_error", "message": str(e)}},
    )
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/ -v --ignore=tests/test_model_resolution.py --ignore=tests/test_dynamic_env_reload.py
```

Expected: all non-ignored tests pass

- [ ] **Step 4: Commit**

```bash
git add responses_api.py
git commit -m "feat: replace local resolver in responses_api with shared model_resolver"
```

---

### Task 7: Update `responses_passthrough.py` — remove local resolver, use shared modules

**Files:**
- Modify: `responses_passthrough.py:1-95`

- [ ] **Step 1: Identify all call sites**

```bash
grep -n "_get_runtime_env_value\|resolve_passthrough_model\|LLM_RESPONSES_MODEL_NAME\|LLM_MODEL_NAME" /Users/yingdong/VSCode/oca_langchain/responses_passthrough.py
```

Note all remaining usages of `_get_runtime_env_value` (for LLM_REASONING_STRENGTH, LLM_NON_REASONING_STRENGTH, etc.) — these must continue to work after removing the local definition.

- [ ] **Step 2: Update imports at top of `responses_passthrough.py`**

Remove:
```python
from dotenv import dotenv_values
_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
def _get_runtime_env_value(key: str, default: str = "") -> str: ...
```

Add:
```python
from runtime_env import _get_runtime_env_value
from model_resolver import resolve_model_for_endpoint
```

- [ ] **Step 3: Replace `resolve_passthrough_model()` call site in `create_response_passthrough()`**

`responses_passthrough.py` does NOT have an existing `_get_chat_model()` helper (unlike `responses_api.py`). Add a new private helper at module level (after imports) and call it at the resolution point.

Add this helper after imports:
```python
def _get_chat_model():
    """Lazy import to avoid circular dependency with api.py."""
    from api import get_chat_model
    return get_chat_model()
```

Find where `resolve_passthrough_model` is called in `create_response_passthrough()` (around line 444). The pattern is:
```python
# Before:
resolved_model = resolve_passthrough_model(original_model)
# ... later:
modified_body["model"] = resolved_model
```

Replace the `resolve_passthrough_model` call:
```python
# After:
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
# Keep the existing line that rewrites the forwarded request body model field:
modified_body["model"] = resolved_model
```

Remove the now-unused `resolve_passthrough_model()` function definition.

- [ ] **Step 4: Run tests**

```bash
pytest tests/ -v --ignore=tests/test_model_resolution.py --ignore=tests/test_dynamic_env_reload.py
```

Expected: all non-ignored tests pass

- [ ] **Step 5: Commit**

```bash
git add responses_passthrough.py
git commit -m "feat: replace local resolver in responses_passthrough with shared model_resolver"
```

---

## Chunk 4: Update existing tests

### Task 8: Rewrite `tests/test_model_resolution.py`

**Files:**
- Modify: `tests/test_model_resolution.py` (full rewrite)

This file currently tests `resolve_model_name()` from `responses_api`. That function is being removed. Replace the entire file with integration tests for the new `resolve_model_for_endpoint()`.

- [ ] **Step 1: Verify existing tests fail (expected)**

```bash
pytest tests/test_model_resolution.py -v
```

Expected: `ImportError: cannot import name 'resolve_model_name' from 'responses_api'`

- [ ] **Step 2: Rewrite the file**

```python
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
```

- [ ] **Step 3: Run the rewritten tests**

```bash
pytest tests/test_model_resolution.py -v
```

Expected: all pass

- [ ] **Step 4: Commit**

```bash
git add tests/test_model_resolution.py
git commit -m "test: rewrite test_model_resolution for resolve_model_for_endpoint"
```

---

### Task 9: Update `tests/test_dynamic_env_reload.py`

**Files:**
- Modify: `tests/test_dynamic_env_reload.py`

Three categories of changes:
1. Remove imports of `resolve_model_name` and `resolve_passthrough_model`
2. Change `_ENV_PATH` patch targets from `responses_api._ENV_PATH` / `responses_passthrough._ENV_PATH` → `runtime_env._ENV_PATH`
3. Update two tests whose expected behavior changes under new resolution logic

- [ ] **Step 1: Verify current failures**

```bash
pytest tests/test_dynamic_env_reload.py -v
```

Expected: import errors on `resolve_model_name` / `resolve_passthrough_model`

- [ ] **Step 2: Update the file**

**Lines 1-2 — Replace imports:**
```python
# Remove:
from responses_api import resolve_model_name
from responses_passthrough import resolve_passthrough_model, resolve_reasoning_effort, resolve_null_reasoning, enforce_pro_model_min_reasoning

# Add:
from model_resolver import resolve_model_for_endpoint, FALLBACK_MODEL
from responses_passthrough import resolve_reasoning_effort, resolve_null_reasoning, enforce_pro_model_min_reasoning
```

**Lines 5-35 — Replace model reload tests (responses_api path):**

The three tests that called `resolve_model_name()` and patched `responses_api._ENV_PATH` must be rewritten. The new function requires `model_api_support` and reads via `runtime_env._ENV_PATH`:

```python
def test_responses_api_model_reload_from_env_each_call(monkeypatch, tmp_path):
    """Env override via LLM_RESPONSES_MODEL_NAME is re-read on each call."""
    env_file = tmp_path / ".env"
    monkeypatch.setattr("runtime_env._ENV_PATH", str(env_file))
    monkeypatch.delenv("LLM_RESPONSES_MODEL_NAME", raising=False)

    catalog = {"oca/gpt-5.1": ["RESPONSES"], "oca/gpt-5.2": ["RESPONSES"]}

    env_file.write_text('LLM_RESPONSES_MODEL_NAME="oca/gpt-5.1"\n', encoding="utf-8")
    assert resolve_model_for_endpoint("gpt-4o", "LLM_RESPONSES_MODEL_NAME", "RESPONSES", catalog) == "oca/gpt-5.1"

    env_file.write_text('LLM_RESPONSES_MODEL_NAME="oca/gpt-5.2"\n', encoding="utf-8")
    assert resolve_model_for_endpoint("gpt-4o", "LLM_RESPONSES_MODEL_NAME", "RESPONSES", catalog) == "oca/gpt-5.2"


def test_responses_api_env_override_takes_priority_over_incoming_oca_model(monkeypatch, tmp_path):
    """When LLM_RESPONSES_MODEL_NAME is set, it takes priority even if incoming has oca/ prefix."""
    env_file = tmp_path / ".env"
    monkeypatch.setattr("runtime_env._ENV_PATH", str(env_file))
    monkeypatch.delenv("LLM_RESPONSES_MODEL_NAME", raising=False)

    catalog = {"oca/gpt-5.9": ["RESPONSES"], "oca/gpt-5.1-codex": ["RESPONSES"]}

    env_file.write_text('LLM_RESPONSES_MODEL_NAME="oca/gpt-5.9"\n', encoding="utf-8")
    # New behavior: env override takes priority, returns oca/gpt-5.9 (not the incoming oca/gpt-5.1-codex)
    assert resolve_model_for_endpoint("oca/gpt-5.1-codex", "LLM_RESPONSES_MODEL_NAME", "RESPONSES", catalog) == "oca/gpt-5.9"


def test_responses_api_model_falls_back_to_fallback_when_env_removed(monkeypatch, tmp_path):
    """When LLM_RESPONSES_MODEL_NAME is removed, resolution falls through to FALLBACK_MODEL if not in catalog."""
    env_file = tmp_path / ".env"
    monkeypatch.setattr("runtime_env._ENV_PATH", str(env_file))
    monkeypatch.delenv("LLM_RESPONSES_MODEL_NAME", raising=False)

    # gpt-4o not in catalog — will fall back to FALLBACK_MODEL
    catalog = {"oca/gpt-5.1": ["RESPONSES"], "oca/gpt-5.4": ["CHAT_COMPLETIONS", "RESPONSES"]}

    env_file.write_text('LLM_RESPONSES_MODEL_NAME="oca/gpt-5.1"\n', encoding="utf-8")
    assert resolve_model_for_endpoint("gpt-4o", "LLM_RESPONSES_MODEL_NAME", "RESPONSES", catalog) == "oca/gpt-5.1"

    # Remove env override — incoming gpt-4o not in catalog → FALLBACK_MODEL
    env_file.write_text("# LLM_RESPONSES_MODEL_NAME removed\n", encoding="utf-8")
    assert resolve_model_for_endpoint("gpt-4o", "LLM_RESPONSES_MODEL_NAME", "RESPONSES", catalog) == FALLBACK_MODEL
```

**Lines 38-71 — Replace passthrough model tests:**

```python
def test_passthrough_model_reload_from_env_each_call(monkeypatch, tmp_path):
    """LLM_RESPONSES_MODEL_NAME is re-read from runtime_env on each call."""
    env_file = tmp_path / ".env"
    monkeypatch.setattr("runtime_env._ENV_PATH", str(env_file))
    monkeypatch.delenv("LLM_RESPONSES_MODEL_NAME", raising=False)

    catalog = {"oca/gpt-5.1": ["RESPONSES"], "oca/gpt-5.2": ["RESPONSES"]}

    env_file.write_text('LLM_RESPONSES_MODEL_NAME="oca/gpt-5.1"\n', encoding="utf-8")
    assert resolve_model_for_endpoint("gpt-4o", "LLM_RESPONSES_MODEL_NAME", "RESPONSES", catalog) == "oca/gpt-5.1"

    env_file.write_text('LLM_RESPONSES_MODEL_NAME="oca/gpt-5.2"\n', encoding="utf-8")
    assert resolve_model_for_endpoint("gpt-4o", "LLM_RESPONSES_MODEL_NAME", "RESPONSES", catalog) == "oca/gpt-5.2"


def test_passthrough_env_override_takes_priority_over_incoming_oca_model(monkeypatch, tmp_path):
    """When LLM_RESPONSES_MODEL_NAME is set, it takes priority even over oca/ incoming model."""
    env_file = tmp_path / ".env"
    monkeypatch.setattr("runtime_env._ENV_PATH", str(env_file))
    monkeypatch.delenv("LLM_RESPONSES_MODEL_NAME", raising=False)

    catalog = {"oca/gpt-5.9": ["RESPONSES"], "oca/gpt-5.1-codex": ["RESPONSES"]}

    env_file.write_text('LLM_RESPONSES_MODEL_NAME="oca/gpt-5.9"\n', encoding="utf-8")
    # New behavior: env override wins
    assert resolve_model_for_endpoint("oca/gpt-5.1-codex", "LLM_RESPONSES_MODEL_NAME", "RESPONSES", catalog) == "oca/gpt-5.9"


def test_passthrough_model_falls_back_when_env_removed(monkeypatch, tmp_path):
    """When LLM_RESPONSES_MODEL_NAME is removed and incoming not in catalog, returns FALLBACK_MODEL."""
    env_file = tmp_path / ".env"
    monkeypatch.setattr("runtime_env._ENV_PATH", str(env_file))
    monkeypatch.delenv("LLM_RESPONSES_MODEL_NAME", raising=False)

    catalog = {"oca/gpt-5.1": ["RESPONSES"], "oca/gpt-5.4": ["CHAT_COMPLETIONS", "RESPONSES"]}

    env_file.write_text('LLM_RESPONSES_MODEL_NAME="oca/gpt-5.1"\n', encoding="utf-8")
    assert resolve_model_for_endpoint("gpt-4o", "LLM_RESPONSES_MODEL_NAME", "RESPONSES", catalog) == "oca/gpt-5.1"

    env_file.write_text("# LLM_RESPONSES_MODEL_NAME removed\n", encoding="utf-8")
    # gpt-4o not in catalog → fallback
    assert resolve_model_for_endpoint("gpt-4o", "LLM_RESPONSES_MODEL_NAME", "RESPONSES", catalog) == FALLBACK_MODEL
```

**Remaining reasoning tests — update `_ENV_PATH` patch target in each of these 4 functions:**

- `test_passthrough_effort_reload_from_env_each_call` (line ~76)
- `test_passthrough_effort_falls_back_to_incoming_when_env_removed` (line ~87)
- `test_passthrough_missing_reasoning_treated_as_null` (line ~101)
- `test_passthrough_missing_reasoning_no_env` (line ~112)

In each of these, change:
```python
# Before:
monkeypatch.setattr("responses_passthrough._ENV_PATH", str(env_file))

# After:
monkeypatch.setattr("runtime_env._ENV_PATH", str(env_file))
```

These tests do not need any other changes — their `resolve_reasoning_effort`, `resolve_null_reasoning` calls are unaffected.

- [ ] **Step 3: Run the updated tests**

```bash
pytest tests/test_dynamic_env_reload.py -v
```

Expected: all pass

- [ ] **Step 4: Run full test suite**

```bash
pytest tests/ -v
```

Expected: all tests pass

- [ ] **Step 5: Commit**

```bash
git add tests/test_dynamic_env_reload.py
git commit -m "test: update test_dynamic_env_reload for new model resolution contract"
```

---

## Final verification

- [ ] **Run full test suite one last time**

```bash
cd /Users/yingdong/VSCode/oca_langchain && source .venv/bin/activate
pytest tests/ -v
```

Expected: all tests pass, no import errors, no regressions

- [ ] **Verify no lingering references to removed functions**

```bash
grep -rn "resolve_model_name\|resolve_passthrough_model\|_get_default_model\|responses_api\._ENV_PATH\|responses_passthrough\._ENV_PATH\|def _get_runtime_env_value" tests/ responses_api.py responses_passthrough.py
```

Expected: no matches

- [ ] **Final commit**

```bash
git add -A
git commit -m "feat: model resolution — endpoint-aware, catalog-backed, unified env reading"
```
