# LangChain Custom LLM & OAuth2 Authentication

This project provides a fully functional custom LangChain `LLM` class and an OpenAI-compatible FastAPI service, enabling you to use local/intranet models as a "self-hosted OpenAI" replacement.

It solves the core problems of integrating private or protected LLM services requiring dynamic token management in the LangChain ecosystem and connecting with multi-end OpenAI SDKs.

## ✨ Main Features

- **Complete OAuth2 Refresh Token Workflow**: Automatically uses a `Refresh Token` to obtain a temporary `Access Token`, supports token rotation, and persists the latest `Refresh Token` to `.env`.
- **Dual API Compatibility**:
  - **OpenAI-Compatible**: `/v1/models` and `/v1/chat/completions` endpoints
  - **Anthropic-Compatible**: `/v1/messages` endpoint (Messages API)
- **Universal SDK Support**: Works with both OpenAI and Anthropic Python/JavaScript SDKs
- **Token Persistence and Caching**: Successfully retrieved `Access Tokens` are written to `.env` and can be reused if not expired after a restart.
- **Seamless LangChain Integration**: Follows `BaseChatModel`, can be called normally via `invoke / stream / astream`.
- **Supports Streaming Responses**: Complete SSE parsing and real-time model output for both API formats.
- **Config Driven**: All sensitive info and model parameters are in `.env`.
- **Async Support**: Both synchronous `requests` and asynchronous `httpx` backend implementations.
- **Intelligent Network Retry & Timeout**: Scenario-adaptive strategy ensures stability and performance, with separate policies for quick operations and long-running inference.

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
│   ├── llm.py               # Core: OCAChatModel
│   └── oauth2_token_manager.py   # OAuth2 token management
├── tests/                   # Test scripts
│   └── test_anthropic_api.sh
├── examples/                # Usage examples
│   └── anthropic_examples.py
├── docs/                    # Documentation
│   └── ANTHROPIC_API.md     # Anthropic API guide
├── run_api.sh                # One-click API service launcher
├── run_ui.sh                 # One-click Streamlit UI launcher
├── .env                      # Environment config (create manually)
├── README.md                 # This file
├── pyproject.toml            # Dependency definitions (uv)
└── uv.lock                   # Locked dependencies
```
## 🚀 Installation & Configuration

**1. Prepare Environment**

This project uses `uv` for package management. It's recommended to work within a virtual environment.

```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment (Linux/macOS)
source .venv/bin/activate

# If you don't have uv, install it
pip install uv
```

**2. Sync Dependencies**

Synchronize all dependencies precisely with the `uv.lock` file for consistent environments.

```bash
uv sync
```

**3. Create and Configure `.env`**

Create a `.env` file based on the following template and fill in your real credentials:

```dotenv
# --- OAuth2 Authentication ---
OAUTH_HOST="your-oauth-host.com"
OAUTH_CLIENT_ID="your_client_id"
OAUTH_REFRESH_TOKEN="your_initial_refresh_token"

# --- LLM API Configuration ---
# Full endpoint URL for your language model API
LLM_API_URL="https://your-llm-api-endpoint/v1/chat/completions"

# Model name to use
LLM_MODEL_NAME="your-model-name"

# API endpoint for available models
LLM_MODELS_API_URL=""

# Default system prompt
LLM_SYSTEM_PROMPT="You are a helpful assistant."

# Default sampling temperature (between 0.0 and 2.0)
LLM_TEMPERATURE="0.7"

# LLM request timeout (in seconds, supports float, e.g. 60s+ recommended)
LLM_REQUEST_TIMEOUT="120"

# --- Network Settings ---
# Force all requests via HTTP proxy (set "true" to always use proxy; case-insensitive)
FORCE_PROXY="false"
# If the app cannot access OAuth or LLM API directly, specify a HTTP proxy
# Example: http://user:password@proxy.example.com:8080
HTTP_PROXY_URL=""

# Additional CA certificates (comma-separated PEM paths; optional)
# MULTI_CA_BUNDLE="./burp_ca.pem,./internal_ca.pem"

# Optional: custom CA bundle path (if set, takes precedence)
# REQUESTS_CA_BUNDLE=/path/to/bundle.pem

# Connection timeout (seconds, supports float, recommended at least 2s; used for quick API requests like token/model fetch.)
CONNECTION_TIMEOUT="2"

# --- Fields below are auto-managed by the program ---
OAUTH_ACCESS_TOKEN=
OAUTH_ACCESS_TOKEN_EXPIRES_AT=
```

### Proxy configuration and runtime debug

- FORCE_PROXY: when set to "true", all OAuth/LLM requests go through the HTTP proxy even if direct access works.
- Auto logic: when FORCE_PROXY is not "true", the manager prefers direct; on failure it retries via proxy and may stick to the last successful mode.
- Runtime changes: the app reloads .env for every request; updating FORCE_PROXY or HTTP_PROXY_URL takes effect immediately without restart.
- HTTPS interception / CA options:
  - Option A (recommended for multiple CAs): set MULTI_CA_BUNDLE to a comma-separated list of PEM files. On startup, the app combines the system CA bundle (certifi) with these extra PEMs and sets REQUESTS_CA_BUNDLE to the combined file (only if REQUESTS_CA_BUNDLE is not already set).
    - Example: MULTI_CA_BUNDLE="./burp_ca.pem,./internal_ca.pem"
    - Note: the combination runs at startup for performance; update MULTI_CA_BUNDLE requires restarting the API server.
  - Option B: set REQUESTS_CA_BUNDLE to an existing full bundle (takes precedence over MULTI_CA_BUNDLE).
    - Example: REQUESTS_CA_BUNDLE=/path/to/ca_with_burp.pem

Typical BURP setup:
1) FORCE_PROXY="true"
2) HTTP_PROXY_URL="http://127.0.0.1:8080"
3) Either:
   - MULTI_CA_BUNDLE="./burp_ca.pem" (auto-combine with system CAs on startup), or
   - REQUESTS_CA_BUNDLE=/absolute/path/to/ca_with_burp.pem (pre-combined bundle)

## 🛠️ How to Run

### Command Line Test

After configuring `.env`, run the original CLI test tool in `core/llm.py`:

```bash
python core/llm.py
```

The script will sequentially demonstrate three usage patterns:
1.  Synchronous streaming (`llm.stream`)
2.  Synchronous non-streaming (`llm.invoke`)
3.  Asynchronous streaming (`llm.astream`)

### Launch the Interactive Chatbot UI

A Streamlit web UI is provided.

**1. Install Streamlit**

```bash
pip install streamlit
```

**2. Start the App**

```bash
streamlit run app.py
```

This will launch a local web server and open the chatbot UI in your browser. You can tune the system prompt, temperature, or pick a model in the sidebar.

## 🤖 Code Overview

### `core/oauth2_token_manager.py`

- **`OCAOauth2TokenManager` Class**:
  - Handles authentication. Completely independent of LangChain, focused on token lifecycle management.
  - On init, tries to load a non-expired `Access Token` from `.env`.
  - `get_access_token()` is the main public method. It checks if the in-memory token is valid, and refreshes if not.
  - `_refresh_tokens()` does OAuth2 communication, swapping `Refresh Token` for a new `Access Token`, handling any new `Refresh Token` in responses, and persists to `.env`.
  - **Network Management**: Includes smart retry logic, switching between direct/proxy for quick operations like model list or refresh; LLM inference uses longer timeout and no retries for output stability.

### `core/llm.py`

- **`OCAChatModel` Class**:
  - Extends LangChain's `BaseChatModel`.
  - Does not handle auth internally; gets an `OCAOauth2TokenManager` instance during init.
  - Before any API call (`_stream`, `_astream`), gets a valid token via `token_manager.get_access_token()`.
  - Implements LangChain standard methods like `_stream` (streaming), `_generate` (non-streaming), and async counterparts.
  - `@classmethod from_env` provides a convenient method to instantiate from env vars.
  - **Dynamic Model List Fetching**: At init, tries to fetch available models via `LLM_MODELS_API_URL`, and supports manual refresh in UI.
  - **Independent Inference Timeout**: LLM inference uses `LLM_REQUEST_TIMEOUT` for long outputs.

### Start the API Server

```bash
# Option 1: Start with uvicorn directly
uvicorn api:app --host 0.0.0.0 --port 8000

# Option 2: Use the one-click script
bash run_api.sh
```
The service listens on port **8000** by default.

**OpenAI-Compatible Endpoints**
| Method | Path | Description        |
|--------|------|-------------------|
| GET    | /v1/models           | Get available model list |
| POST   | /v1/chat/completions | Chat completion (supports `stream=true`) |

**Anthropic-Compatible Endpoints**
| Method | Path | Description        |
|--------|------|-------------------|
| POST   | /v1/messages | Messages API completion (supports `stream=true`) |

> **Note**: Both endpoints share the same backend infrastructure and OAuth2 authentication.

**Quick Usage Example**

Non-streaming:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model":"your-model-name",
        "messages":[{"role":"user","content":"Hello!"}]
      }'
```
Streaming SSE:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model":"your-model-name",
        "messages":[{"role":"user","content":"Tell me a joke"}],
        "stream":true
      }'
```
Python:
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="any")

response = client.chat.completions.create(
    model="your-model-name",
    messages=[{"role": "user", "content": "Hello!"}]
)

print(response.choices[0].message.content)
```

## 🔷 Anthropic Messages API

The service also provides compatibility with Anthropic's Messages API, allowing you to use the official Anthropic SDK with your custom backend.

### Quick Start

Install the Anthropic SDK:
```bash
pip install anthropic
```

Basic usage:
```python
import anthropic

client = anthropic.Anthropic(
    base_url="http://localhost:8000",
    api_key="test"  # Optional
)

message = client.messages.create(
    model="your-model-name",
    max_tokens=100,
    messages=[{"role": "user", "content": "Hello!"}]
)

print(message.content[0].text)
```

### Anthropic-Specific Features

- **Tool/Function Calling**: Use Anthropic's tool definition format
- **Streaming Events**: Full SSE support with 6 event types
- **Multipart Content**: Text + tool_use + tool_result in single response

**Curl Example (Anthropic Format):**
```bash
curl http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: test" \
  -d '{
    "model": "your-model-name",
    "max_tokens": 100,
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

For detailed documentation, see [docs/ANTHROPIC_API.md](docs/ANTHROPIC_API.md)

### Testing

**Test OpenAI endpoint:**
```bash
bash tests/test_anthropic_api.sh
```

**Run Python examples:**
```bash
python examples/anthropic_examples.py
```

