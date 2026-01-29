# LangChain Custom LLM & OAuth2 Authentication

This project provides a fully functional custom LangChain `LLM` class and an OpenAI-compatible FastAPI service, enabling you to use local/intranet models as a "self-hosted OpenAI" replacement. It also offers an Anthropic Messages API compatibility layer so Anthropic SDKs can talk to the same backend.

It solves the core problems of integrating private or protected LLM services requiring dynamic token management in the LangChain ecosystem and connecting with multi-end OpenAI/Anthropic SDKs.

## 🚀 What’s New (Latest Updates)

Based on recent changes (see git history), the project has gained several important capabilities:

- Unified, weight-based message validation for tool calls
  - Prevents 422/format errors by healing incomplete tool_call sequences and orphaned tool_result messages
  - Works for both streaming and non-streaming responses
  - Implemented in `core/llm.py` and applied automatically for API calls
- Anthropic streaming tool_calls ↔ tool_use conversion
  - End-to-end support for converting streaming deltas between OpenAI-style `tool_calls` and Anthropic `tool_use` blocks
  - Corrects tool_result → OpenAI ToolMessage mapping
- New endpoints for wider compatibility
  - `GET /v1/model/info` (LiteLLM-compatible model info endpoint)
  - `POST /v1/spend/calculate` (simple spend calculation endpoint)
- Improved validation, testing, and debugging artifacts
  - Extensive tests for edge cases and unified validation, including `test_unified_validation.py`
  - New docs: `TESTING.md`, `TEST_RESULTS.md`, `DEBUGGING_NOTES.md`
- Network and security enhancements
  - Optional SSL verification bypass for proxy/MITM debugging: `DISABLE_SSL_VERIFY="true"`
  - Multi-CA bundle merge via `MULTI_CA_BUNDLE` for BURP/internal CA scenarios
- Logging improvements
  - Centralized logging with `LOG_LEVEL` and `LOG_FILE_PATH`

Jump to sections below for details about configuration, endpoints, tool-call validation, and Anthropic compatibility.

## ✨ Main Features

- Complete OAuth2 Refresh Token Workflow: automatically obtains/rotates `Access Tokens`, persists updated tokens to `.env`
- Dual API Compatibility:
  - OpenAI-Compatible: `/v1/models` and `/v1/chat/completions`
  - Anthropic-Compatible: `/v1/messages` (Messages API)
- Universal SDK Support: Works with both OpenAI and Anthropic Python/JavaScript SDKs
- Token Persistence and Caching: reuses non-expired `Access Tokens` across restarts
- Seamless LangChain Integration: Implements `BaseChatModel` with `invoke / stream / astream`
- Streaming Responses: Complete SSE parsing for both API formats
- Config Driven: All sensitive info and model parameters are in `.env`
- Async Support: Synchronous `requests` and asynchronous `httpx` backends
- Intelligent Network Retry & Timeout: Different strategies for quick operations vs. long-running inference

## 📂 Project Structure
```
.
├── app.py                    # Streamlit chatbot UI
├── api.py                    # FastAPI service (OpenAI + Anthropic compatible)
├── anthropic_api.py          # Anthropic Messages API endpoints
├── models/                   # API type definitions
│   └── anthropic_types.py    # Anthropic Pydantic models
├── converters/               # Format converters
│   └── anthropic_request_converter.py  # Anthropic ↔ LangChain
├── core/
│   ├── llm.py                # Core: OCAChatModel + unified tool-call validation
│   └── oauth2_token_manager.py   # OAuth2 token management
├── tests/                    # Test scripts and helpers
│   ├── test_422_debug.py
│   ├── test_422_request_validation.py
│   ├── test_tool_result_api.py
│   ├── test_tool_result_conversion.py
│   ├── test_unified_validation.py
│   └── test_anthropic_api.sh
├── examples/
│   └── anthropic_examples.py
├── docs/
│   └── ANTHROPIC_API.md
├── run_api.sh                # One-click API service launcher (defaults to port 8450)
├── run_ui.sh                 # One-click Streamlit UI launcher
├── .env                      # Environment config (create manually)
├── README.md                 # This file
├── pyproject.toml            # Dependencies (uv)
└── uv.lock                   # Locked dependencies
```

## ⚙️ Installation & Configuration

1) Prepare environment

```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment (Linux/macOS)
source .venv/bin/activate

# If you don't have uv, install it
pip install uv
```

2) Sync dependencies

```bash
uv sync
```

3) Create and configure `.env`

```dotenv
# --- OAuth2 Authentication ---
OAUTH_HOST="your-oauth-host.com"
OAUTH_CLIENT_ID="your_client_id"
OAUTH_REFRESH_TOKEN="your_initial_refresh_token"

# --- LLM API Configuration ---
LLM_API_URL="https://your-llm-api-endpoint/v1/chat/completions"
LLM_MODEL_NAME="your-model-name"
LLM_MODELS_API_URL=""
LLM_SYSTEM_PROMPT="You are a helpful assistant."
LLM_TEMPERATURE="0.7"

# --- Network Settings ---
FORCE_PROXY="false"
HTTP_PROXY_URL=""
# Merge extra PEMs into system CAs at startup (comma-separated)
MULTI_CA_BUNDLE=""
# Disable SSL verify globally (debug only)
DISABLE_SSL_VERIFY="false"
# Optional: provide full CA bundle instead (takes precedence)
# REQUESTS_CA_BUNDLE=/absolute/path/to/ca_with_burp.pem

# Quick operations timeout (seconds, e.g., token refresh/model list)
CONNECTION_TIMEOUT="2"
# Inference timeout (seconds, long-running requests)
LLM_REQUEST_TIMEOUT="180"

# --- Logging ---
LOG_LEVEL="INFO"     # DEBUG or INFO
LOG_FILE_PATH="logs/llm_api.log"

# --- Auto-managed ---
OAUTH_ACCESS_TOKEN=
OAUTH_ACCESS_TOKEN_EXPIRES_AT=
```

### Proxy configuration and runtime debug

- FORCE_PROXY: when set to "true", all OAuth/LLM requests go through the HTTP proxy even if direct access works.
- Auto logic: when FORCE_PROXY is not "true", the manager prefers direct; on failure it retries via proxy and may stick to the last successful mode.
- Runtime changes: the app reloads .env for every request; updating FORCE_PROXY or HTTP_PROXY_URL takes effect immediately without restart.
- HTTPS interception / CA options:
  - Option A: set MULTI_CA_BUNDLE to a comma-separated list of PEM files. On startup, the app combines system CAs with these extra PEMs and sets REQUESTS_CA_BUNDLE to the combined file (unless REQUESTS_CA_BUNDLE is already set).
  - Option B: set REQUESTS_CA_BUNDLE to an existing full bundle (takes precedence over MULTI_CA_BUNDLE).
- Debug-only: set DISABLE_SSL_VERIFY="true" to bypass SSL verification (use only for local MITM/proxy debugging).

Typical BURP setup:
1) FORCE_PROXY="true"
2) HTTP_PROXY_URL="http://127.0.0.1:8080"
3) Either MULTI_CA_BUNDLE="./burp_ca.pem" or REQUESTS_CA_BUNDLE=/absolute/path/to/ca_with_burp.pem

## 🛠️ How to Run

### Command Line Test

```bash
python core/llm.py
```
The script demonstrates:
1) Synchronous streaming (`llm.stream`)
2) Synchronous non-streaming (`llm.invoke`)
3) Asynchronous streaming (`llm.astream`)

### Launch the Interactive Chatbot UI

```bash
pip install streamlit
streamlit run app.py
```

### Start the API Server

```bash
# Option 1: Start with uvicorn directly (port 8450 for consistency)
uvicorn api:app --host 0.0.0.0 --port 8450

# Option 2: Use the one-click script (also 8450)
bash run_api.sh
```

## 🔌 HTTP Endpoints

OpenAI-Compatible Endpoints
- GET  /v1/models                — Get available model list
- POST /v1/chat/completions      — Chat completion (supports `stream=true`)
- GET  /v1/model/info            — LiteLLM-compatible models info
- POST /v1/spend/calculate       — Simple spend calculation response

Anthropic-Compatible Endpoints (Messages API)
- POST /v1/messages              — Completion (supports `stream=true`)

Note: Both formats share the same backend and OAuth2 auth.

### Quick Usage Examples

OpenAI non-streaming:
```bash
curl http://localhost:8450/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model":"your-model-name",
        "messages":[{"role":"user","content":"Hello!"}]
      }'
```

OpenAI streaming SSE:
```bash
curl http://localhost:8450/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model":"your-model-name",
        "messages":[{"role":"user","content":"Tell me a joke"}],
        "stream":true
      }'
```

OpenAI Python SDK:
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8450/v1", api_key="any")

response = client.chat.completions.create(
    model="your-model-name",
    messages=[{"role": "user", "content": "Hello!"}]
)

print(response.choices[0].message.content)
```

Anthropic Messages API (Python):
```python
import anthropic

client = anthropic.Anthropic(
    base_url="http://localhost:8450",
    api_key="test"  # Optional
)

message = client.messages.create(
    model="your-model-name",
    max_tokens=100,
    messages=[{"role": "user", "content": "Hello!"}]
)

print(message.content[0].text)
```

## 🧩 Tool Calls and Unified Validation

The model supports tool/function calling for both OpenAI and Anthropic formats. To ensure robust behavior across providers and transport modes, a unified, weight-based validation algorithm is used:

- Message weighting
  - +N: Assistant message with `tool_calls` (weight equals number of tool calls)
  - -1: Tool (tool_result) message
  - 0 : All other messages
- Validation logic (high level)
  - Scan messages while tracking how many tool results are expected vs. seen
  - Handle streaming and non-streaming sequences consistently
  - If an interrupting message appears before all required tool results are present, drop the incomplete `tool_calls` from the assistant message and skip orphaned `tool_result` blocks

This prevents malformed transcripts from reaching the backend and avoids provider errors (e.g., 422). The algorithm is applied transparently by the API layer.

## 🔷 Anthropic Compatibility Details

- `tool_use` and `tool_result` conversion
  - Assistant `tool_use` blocks ↔ OpenAI `tool_calls` (function type)
  - User `tool_result` blocks → OpenAI ToolMessage with `tool_call_id`
- Streaming support
  - Converts streaming deltas for tool uses/results to/from OpenAI-style incremental `tool_calls` updates
- Mixed content handling
  - Correctly separates text vs. tool blocks, including messages that include both text and tool results

See `docs/ANTHROPIC_API.md` and `examples/anthropic_examples.py` for end-to-end usage in Anthropic format.

## 📜 Logging & Debugging

- Configure verbosity via `LOG_LEVEL` (DEBUG/INFO)
- Logs written to `LOG_FILE_PATH` (e.g., `logs/llm_api.log`)
- Requests/responses are logged with Authorization headers redacted
- Tool calls and streaming deltas are captured for easier debugging

## ✅ Testing

Run the provided tests and scripts to validate behavior:

- Shell: `bash tests/test_anthropic_api.sh`
- Python tests (examples):
  - `tests/test_422_debug.py`
  - `tests/test_422_request_validation.py`
  - `tests/test_tool_result_api.py`
  - `tests/test_tool_result_conversion.py`
  - `tests/test_unified_validation.py`

Documentation notes and results:
- `TESTING.md` — how tests are organized and run
- `TEST_RESULTS.md` — sample outputs and verified scenarios
- `DEBUGGING_NOTES.md` — notes captured during debugging sessions

## 🤖 Code Overview

### core/oauth2_token_manager.py

- `OCAOauth2TokenManager` manages OAuth2 lifecycle and network behavior
- Direct/proxy switching with retries for quick ops (refresh/model list)
- Honors CA settings (`MULTI_CA_BUNDLE` / `REQUESTS_CA_BUNDLE`) and `DISABLE_SSL_VERIFY`
- Persists updated tokens to `.env`

### core/llm.py

- `OCAChatModel` implements LangChain `BaseChatModel`
- Uses `token_manager.get_access_token()` before requests
- Implements `_stream`, `_generate`, and async counterparts
- Applies unified tool-call validation before payload build
- Reconstructs final tool_calls list from streaming deltas for logging and response formatting

## 📌 Notes

- Python >= 3.13
- Start API on port 8450 (via script) or choose your own port with uvicorn
- Both OpenAI and Anthropic SDKs can target this server; choose the endpoint format that best matches your client
