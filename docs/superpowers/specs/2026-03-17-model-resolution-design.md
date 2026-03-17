# Model Resolution Logic Design

**Date**: 2026-03-17
**Status**: Approved
**Scope**: `/v1/chat/completions` and `/v1/responses` endpoints

---

## Problem

Clients (e.g., Codex CLI) send requests with model names that lack the `oca/` prefix (e.g., `gpt-5-codex`, `gpt-4.1`). Currently:

- Chat completions: returns 404 immediately if model not in `available_models`
- Responses API: falls back to `LLM_MODEL_NAME` from env (wrong env key), or uses incoming model as-is (also 404)
- Passthrough: blindly prefixes `oca/` without validating endpoint support

Additionally, the backend distinguishes models by supported endpoint type — not all models support both `CHAT_COMPLETIONS` and `RESPONSES`. This information was not captured at startup and not used in routing.

## Goals

1. When `LLM_MODEL_NAME` (chat) or `LLM_RESPONSES_MODEL_NAME` (responses) is not set, automatically resolve `gpt-x` → `oca/gpt-x` if that model supports the endpoint
2. Fall back to `oca/gpt-5.4` when no match is found
3. Cache the model catalog (including `supported_api_list`) at startup — no per-request API calls
4. Unify the three duplicated `_get_runtime_env_value()` implementations
5. When an env override is explicitly set but the model doesn't support the endpoint, return HTTP 500 with a diagnostic error (explicit misconfiguration should be visible)

## Non-Goals

- Changing the Anthropic `/v1/messages` endpoint behavior
- Adding per-request model catalog refresh
- UI/Streamlit changes
- Fixing the pre-existing `chat_model.model = request.model` concurrency issue (see Known Limitations)

---

## Design

### New Files (root level)

#### `runtime_env.py`

Centralizes the dynamic `.env` reading logic currently duplicated in `responses_api.py` and `responses_passthrough.py`.

```python
_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")

def _get_runtime_env_value(key: str, default: str = "") -> str:
    """Read env value preferring .env file over os.environ for runtime reload support."""
```

#### `model_resolver.py`

Pure function for model resolution. Accepts `model_api_support` as a parameter (no internal global state, fully testable).

```python
FALLBACK_MODEL = "oca/gpt-5.4"
# oca/gpt-5.4 is documented to support both CHAT_COMPLETIONS and RESPONSES.
# At startup, fetch_available_models() logs a WARNING if FALLBACK_MODEL is not in
# the loaded catalog. If the backend later removes it, requests hitting the fallback
# path will fail downstream with a clear "model not found" error.

def resolve_model_for_endpoint(
    incoming_model: str,
    env_key: str,
    endpoint_type: str,
    model_api_support: Dict[str, List[str]],
) -> str:
    """
    Resolve which model to use for a given endpoint type.

    Args:
        incoming_model: Model name from the request (may or may not have oca/ prefix)
        env_key: Env var to check for operator override ("LLM_MODEL_NAME" or "LLM_RESPONSES_MODEL_NAME")
        endpoint_type: Required capability ("CHAT_COMPLETIONS" or "RESPONSES")
        model_api_support: Dict mapping model_id -> list of supported endpoint types (uppercase).
                           Empty dict means catalog was not loaded at startup (fail-open mode).

    Returns:
        Resolved model name (always non-empty)

    Raises:
        ValueError: If env override is set, catalog is non-empty, and the model does not
                    support the requested endpoint type (explicit misconfiguration).
    """
```

**Resolution logic (in order):**

1. **Normalize incoming**: `strip()`. If `None`/empty after strip → skip to step 5 (fallback)

2. **Env override present** (read `env_key` from `.env` via `_get_runtime_env_value`):
   - Normalize the env override value with `oca/` prefix if missing (same normalization as step 3 — operators may configure with or without the prefix)
   - If set AND `model_api_support` is non-empty (catalog loaded successfully):
     - Normalized model ID not found in catalog → **raise `ValueError`**: `"Configured {env_key}='{raw_value}' (resolved to '{normalized}') is not in the model catalog. Check your .env."`
     - Model found, supports `endpoint_type` → **use normalized env model**
     - Model found, does NOT support `endpoint_type` → **raise `ValueError`**: `"Configured {env_key}='{raw_value}' does not support {endpoint_type}. Supported endpoints for this model: {list}. Check your .env."`
   - If set AND `model_api_support` is empty (catalog fetch failed at startup):
     - Log warning: `"Model catalog unavailable; using configured {env_key}='{normalized}' without endpoint validation"`
     - **Use normalized env model** (fail-open — we cannot validate without catalog)
   - If not set → continue to step 3

3. **Normalize incoming with `oca/` prefix** if not already present
   - Strip extra `oca/` prefixes to prevent `oca/oca/...` double-prefix

4. **Check catalog** (only if `model_api_support` is non-empty — catalog was loaded):
   - Look up `model_api_support.get(candidate)`:
     - Returns `None` (model not in catalog) → skip to step 5
     - Returns `[]` or model has no `supported_api_list` entry: treat as **supports all endpoints** (backward compatibility with old catalog format that lacked this field) → **use candidate**
     - Returns non-empty list: if `endpoint_type` in list → **use candidate**; else → skip to step 5

   If `model_api_support` is empty (`{}`) — this covers all of: API not configured, fetch failed, or backend returned empty model list. All three are treated identically as "catalog unavailable" and use the same fail-open behavior. The distinction between "fetch failed" and "empty list" is a diagnostic concern logged at startup, not a routing concern:
   - Fail-open: **use `candidate`** with warning log `"Model catalog unavailable; using '{candidate}' without endpoint validation for {endpoint_type}"`

5. **Fallback**: return `FALLBACK_MODEL` (`oca/gpt-5.4`) + log warning including `incoming_model` and `endpoint_type`

**Edge cases handled:**

| Input | Behavior |
|-------|----------|
| `None` / `""` / whitespace | Skip to fallback |
| `"oca/gpt-4.1"` (already prefixed) | Validate against catalog; use if supported, else fallback |
| `"oca/oca/gpt-5.4"` (double prefix) | Normalize: strip extra prefix, then validate |
| `supported_api_list == []` or field missing | Treat as "supports all endpoints" (backward compatibility) |
| `supported_api_list` mixed case | Stored as uppercase at fetch time; comparisons are uppercase |
| `model_api_support == {}` (catalog empty/unavailable), no env override | Fail-open: use `oca/{incoming}` with warning, no endpoint validation |
| env override without `oca/` prefix (e.g., `LLM_MODEL_NAME=gpt-4.1`) | Normalized to `oca/gpt-4.1` then validated |
| env override + empty catalog | Warn, use normalized env model (catalog fetch may have failed at startup) |
| env override → model not in catalog | `ValueError` → HTTP 500 |
| env override → model in catalog, wrong endpoint | `ValueError` → HTTP 500 with supported endpoint list |
| No env override, model not supported | Silent fallback to `oca/gpt-5.4` with warning log |

---

### Modified: `core/llm.py` — OCAChatModel

**Field changes:**
```python
# Before (mutable default — works but fragile with Pydantic compat layers)
available_models: List[str] = []

# After (explicit default_factory)
available_models: List[str] = Field(default_factory=list)
model_api_support: Dict[str, List[str]] = Field(default_factory=dict)
```

Fixes existing mutable-default issue on `available_models` (Pydantic v1/v2 / LangChain BaseChatModel compatibility).

**`fetch_available_models()` — atomic update via temp snapshots:**

To prevent partial state if the fetch fails mid-way, populate temporary variables first and only assign to instance fields on full success. On failure, prior cached values are preserved unchanged.

```python
def fetch_available_models(self):
    if not self.models_api_url:
        ...
        return

    # Use temp snapshots — only commit to self.* on success
    # This preserves prior cache on failure and prevents partial updates
    new_available = []
    new_api_support = {}

    # ... HTTP request + response.raise_for_status() ...

    for model in models_data:
        model_id = model.get("id") or model.get("litellm_params", {}).get("model")
        if model_id:
            new_available.append(model_id)
            # NEW: capture supported endpoint types (normalized to uppercase)
            supported = model.get("model_info", {}).get("supported_api_list") or []
            new_api_support[model_id] = [s.upper() for s in supported]

    # Atomic commit: both fields replaced together only on success
    self.available_models = new_available
    self.model_api_support = new_api_support

    # Startup validation: warn if fallback model has issues
    if new_available:
        fallback_support = new_api_support.get(FALLBACK_MODEL)
        if fallback_support is None:
            logger.warning(f"Fallback model '{FALLBACK_MODEL}' is not in the model catalog. "
                           f"Requests routed to it will fail downstream.")
        else:
            # Use two separate ifs to capture both warnings independently
            if fallback_support and "CHAT_COMPLETIONS" not in fallback_support:
                logger.warning(f"Fallback model '{FALLBACK_MODEL}' does not support CHAT_COMPLETIONS.")
            if fallback_support and "RESPONSES" not in fallback_support:
                logger.warning(f"Fallback model '{FALLBACK_MODEL}' does not support RESPONSES.")
        # Note: fallback_support == [] means "supports all" per backward compat rule — no warning needed
    return

    # Exception paths: self.* is NOT touched — prior values are preserved
```

On fetch failure: both `available_models` and `model_api_support` keep their prior values (or `[]`/`{}` on first call). `model_api_support == {}` is the "catalog unavailable" signal used by `resolve_model_for_endpoint` (fail-open mode — incoming model used without endpoint validation).

---

### Modified: `api.py` — `/v1/chat/completions`

Replace the current "404 if model not in available_models" guard with model resolution:

```python
from model_resolver import resolve_model_for_endpoint

# In create_chat_completion():
try:
    resolved_model = resolve_model_for_endpoint(
        request.model,
        "LLM_MODEL_NAME",
        "CHAT_COMPLETIONS",
        chat_model.model_api_support,
    )
except ValueError as e:
    raise HTTPException(status_code=500, detail=str(e))

# Update both chat_model.model (used by LLM call) AND request.model (used in response body)
# so the client sees the resolved model name in the response, not the original incoming name
chat_model.model = resolved_model
request.model = resolved_model
```

The resolved model replaces `request.model` for the duration of the request. The existing `chat_model.model = request.model` assignment pattern is preserved as-is (not changed by this feature).

---

### Modified: `responses_api.py`

- Remove local `_get_runtime_env_value()` → import from `runtime_env`
- Remove `resolve_model_name()` → replace with `resolve_model_for_endpoint()`
- Change env key from `LLM_MODEL_NAME` → `LLM_RESPONSES_MODEL_NAME`

```python
from runtime_env import _get_runtime_env_value
from model_resolver import resolve_model_for_endpoint

# In create_response():
try:
    request.model = resolve_model_for_endpoint(
        request.model,
        "LLM_RESPONSES_MODEL_NAME",
        "RESPONSES",
        chat_model.model_api_support,
    )
except ValueError as e:
    raise HTTPException(status_code=500, detail={"type": "error", "error": {"type": "configuration_error", "message": str(e)}})
```

---

### Modified: `responses_passthrough.py`

- Remove local `_get_runtime_env_value()` → import from `runtime_env`
- Replace `resolve_passthrough_model()` body with `resolve_model_for_endpoint()`
- Get `model_api_support` via lazy `_get_chat_model()` import (existing pattern in `responses_api.py`)
- Dependency failure handling: if `_get_chat_model()` raises (chat model not initialized), propagate as HTTP 500 — same behavior as today for all other routes that require the chat model

```python
# In create_response_passthrough():
from api import get_chat_model  # lazy import (existing pattern)
try:
    chat_model = get_chat_model()  # raises HTTPException 500 if not initialized
    incoming = request_body.get("model", "")
    resolved = resolve_model_for_endpoint(
        incoming,
        "LLM_RESPONSES_MODEL_NAME",
        "RESPONSES",
        chat_model.model_api_support,
    )
except ValueError as e:
    # Use Response API error envelope format (same as responses_api.py)
    raise HTTPException(status_code=500, detail={"type": "error", "error": {"type": "configuration_error", "message": str(e)}})
request_body["model"] = resolved
```

---

## Data Flow

```
Request arrives (model="gpt-4.1")
        │
        ▼
resolve_model_for_endpoint(
    incoming="gpt-4.1",
    env_key="LLM_MODEL_NAME",
    endpoint_type="CHAT_COMPLETIONS",
    model_api_support={
        "oca/gpt-4.1": ["RESPONSES", "CHAT_COMPLETIONS"],
        "oca/gpt-5-codex": ["RESPONSES"],
        ...
    }
)
        │
        ├─ LLM_MODEL_NAME set? No
        │
        ├─ Normalize: "gpt-4.1" → "oca/gpt-4.1"
        │
        ├─ "CHAT_COMPLETIONS" in ["RESPONSES", "CHAT_COMPLETIONS"]? Yes
        │
        └─ Return "oca/gpt-4.1"
```

---

## Known Limitations (Pre-existing, Out of Scope)

**Shared `chat_model` instance mutation**: `api.py` currently assigns `chat_model.model = request.model` on a shared singleton, which is a potential race condition under concurrent requests. This design preserves that existing pattern without making it worse. Fixing it would require passing the model name explicitly into `_stream`/`_generate` rather than storing it on the instance — that is out of scope for this feature.

**`fetch_available_models()` duplicate-accumulation**: Multiple calls (e.g., Streamlit UI reload) previously accumulated duplicates in `available_models`. This design fixes it via the temp-snapshot pattern: new lists are built fresh on each call and only assigned to `self.*` on success — so repeated calls start with clean temp state without resetting the live cache prematurely.

---

## Testing

### Unit tests for `model_resolver.resolve_model_for_endpoint()`

All branches of the resolution logic:
- `None` / empty / whitespace incoming → returns fallback
- Incoming already has `oca/` prefix + supported → uses it
- Incoming already has `oca/` prefix + unsupported → returns fallback
- Incoming without prefix + supported after prefix → uses prefixed
- Incoming without prefix + unsupported → returns fallback
- `oca/oca/...` double prefix → normalized and validated
- `model_api_support[model] == []` (empty list) → treated as "supports all endpoints" (backward compat)
- `model_api_support[model]` key absent (model not in catalog) → treated as unsupported, fallback
- Env override set + catalog non-empty + model supports endpoint → uses env model
- Env override set + catalog non-empty + model NOT in catalog → `ValueError` with "not in catalog" message
- Env override set + catalog non-empty + model in catalog, wrong endpoint → `ValueError` with supported list
- Env override set + catalog empty → warns + uses env model
- Env override not set + no catalog match → returns `FALLBACK_MODEL`
- Env override not set + catalog empty (`{}`) + incoming non-empty → fail-open, return `oca/{incoming}`
- Env override not set + catalog non-empty + model has `supported_api_list=[]` → treated as supports all, use model
- Env override not set + catalog non-empty + model has no `supported_api_list` key → treated as supports all, use model

### Unit tests for `core/llm.py`

- `fetch_available_models()` populates `model_api_support` with uppercase values
- Model with no `supported_api_list` in catalog → stored as `[]`, treated as "supports all"
- Repeated `fetch_available_models()` calls do not duplicate entries (idempotent)
- First call + fetch failure → `available_models` and `model_api_support` remain `[]`/`{}`
- Second call (after successful first) + fetch failure → prior cached values are preserved (not cleared)
- Catalog loaded + `FALLBACK_MODEL` absent → warning logged
- Catalog loaded + `FALLBACK_MODEL` present but doesn't support `CHAT_COMPLETIONS` → warning logged
- Catalog loaded + `FALLBACK_MODEL` present but doesn't support `RESPONSES` → warning logged
- Catalog loaded + `FALLBACK_MODEL` has `supported_api_list == []` → no warning (treated as supports all)

### Integration tests (existing `tests/` directory)

**Chat completions (`/v1/chat/completions`):**
- Model `"gpt-4.1"` (no prefix) → resolves to `"oca/gpt-4.1"`, response body `model` field also shows `"oca/gpt-4.1"` (not original `"gpt-4.1"`)
- Model `"gpt-5-codex"` (RESPONSES-only) → fallback to `oca/gpt-5.4`
- `LLM_MODEL_NAME=oca/gpt-5-codex` → HTTP 500 (misconfiguration)
- `LLM_MODEL_NAME=gpt-4.1` (without oca/ prefix) → normalized to `oca/gpt-4.1`, used if supported

**Responses API (`/v1/responses`, LangChain path):**
- Model `"gpt-4.1"` (no prefix) → resolves to `"oca/gpt-4.1"`
- Model `"gpt-oss-120b"` (CHAT_COMPLETIONS-only) → fallback to `oca/gpt-5.4`
- `LLM_RESPONSES_MODEL_NAME=oca/gpt-oss-120b` → HTTP 500 with Response API error envelope

**Passthrough (`/v1/responses` with `LLM_RESPONSES_API_URL` set):**
- Model `"gpt-4.1"` → request body `model` is rewritten to `"oca/gpt-4.1"`
- `LLM_RESPONSES_MODEL_NAME=oca/gpt-oss-120b` → HTTP 500 with Response API error envelope (same schema as LangChain path)
- Chat model not initialized → HTTP 500 propagated from `get_chat_model()`
- Catalog empty + env override set → warn + use env override model (fail-open)

**Startup resilience:**
- Catalog fetch failed at startup → env override used without endpoint validation (warn only)

---

## Codex Discussion Consensus (2026-03-17, 2 rounds)

- **Decision**: Two new root-level modules (`runtime_env.py` + `model_resolver.py`) + OCAChatModel extension + three call-site updates
- **My position**: Single `model_resolver.py`; env override warn + fallback on misconfiguration
- **Codex's position**: Split files; HTTP 500 on env misconfiguration; validate all models (with or without oca/ prefix) against catalog
- **Resolution**: Full convergence — two-file split adopted; HTTP 500 for misconfigured env override with distinct error messages for "not in catalog" vs "wrong endpoint"; all models validated against catalog
- **Reasoning**: Env override keys are endpoint-specific operator choices; silent fallback would mask misconfiguration. File split matches single-responsibility principle and eliminates duplicate `_get_runtime_env_value` code.
- **Rejected alternatives**: Single `model_resolver.py` (mixes concerns), `core/model_resolver.py` (wrong `.env` path, wrong layer), OCAChatModel resolve method (mixes LLM layer with endpoint routing concerns), warn+fallback on env override (hides operator mistakes)
