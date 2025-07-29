# LangChain Custom LLM & OAuth2 Authentication

This project provides a fully functional custom LangChain `LLM` class and an OpenAI-compatible FastAPI service, enabling you to use local/intranet models as a "self-hosted OpenAI" replacement.

It solves the core problems of integrating private or protected LLM services requiring dynamic token management in the LangChain ecosystem and connecting with multi-end OpenAI SDKs.

## ✨ Main Features

- **Complete OAuth2 Refresh Token Workflow**: Automatically uses a `Refresh Token` to obtain a temporary `Access Token`, supports token rotation, and persists the latest `Refresh Token` to `.env`.
- **OpenAI-Compatible FastAPI Service**: Offers `/v1/models` and `/v1/chat/completions` endpoints, allows direct OpenAI API replacement, supporting both streaming & non-streaming.
- **Token Persistence and Caching**: Successfully retrieved `Access Tokens` are written to `.env` and can be reused if not expired after a restart.
- **Seamless LangChain Integration**: Follows `BaseChatModel`, can be called normally via `invoke / stream / astream`.
- **Supports Streaming Responses**: Complete SSE parsing and real-time model output.
- **Config Driven**: All sensitive info and model parameters are in `.env`.
- **Async Support**: Both synchronous `requests` and asynchronous `httpx` backend implementations.
- **Intelligent Network Retry & Timeout**: Scenario-adaptive strategy ensures stability and performance, with separate policies for quick operations and long-running inference.

## 📂 Project Structure
```
.
├── app.py                    # Streamlit chatbot UI
├── api.py                    # OpenAI-compatible FastAPI service
├── core/llm.py               # Core: OCAChatModel
├── core/oauth2_token_manager.py   # OAuth2 token management
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
# If the app cannot access OAuth or LLM API directly, specify a HTTP proxy
# Example: http://user:password@proxy.example.com:8080
HTTP_PROXY_URL=""

# Connection timeout (seconds, supports float, recommended at least 2s; used for quick API requests like token/model fetch.)
CONNECTION_TIMEOUT="2"

# --- Fields below are auto-managed by the program ---
OAUTH_ACCESS_TOKEN=
OAUTH_ACCESS_TOKEN_EXPIRES_AT=
```

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

### Start the OpenAI-Compatible API Server

```bash
# Option 1: Start with uvicorn directly
uvicorn api:app --host 0.0.0.0 --port 8000

# Option 2: Use the one-click script
bash run_api.sh
```
The service listens on port **8000** by default.

**Main Endpoints**
| Method | Path | Description        |
|--------|------|-------------------|
| GET    | /v1/models           | Get available model list |
| POST   | /v1/chat/completions | Chat completion (supports `stream=true`) |

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
