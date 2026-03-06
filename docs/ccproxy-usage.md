# Using ccproxy with Claude Code

This guide explains how to use ccproxy to proxy Claude Code requests to the local oca_api service.

## Architecture

```
Claude Code → ccproxy (localhost:4000) → oca_api (localhost:8450) → Backend LLM
```

## Quick Start

1. Start oca_api:
   ```bash
   cd /Users/yingdong/VSCode/oca_langchain
   bash run_api.sh
   ```

2. Start ccproxy:
   ```bash
   ccproxy start --detach
   ```

3. Reload shell or source ~/.zshrc:
   ```bash
   source ~/.zshrc
   ```

4. Run Claude Code as usual

## Model Mapping

| Claude Model | Target Model | Endpoint |
|--------------|--------------|----------|
| `claude-opus-4-6` | `oca/gpt-5.2` | `/v1/chat/completions` |
| `claude-sonnet-4-6` | `oca/gpt-5.2` | `/v1/chat/completions` |
| `claude-haiku-4-5-20251001` | `oca/llama4` | `/v1/chat/completions` |
| `claude-haiku-4-5` | `oca/llama4` | `/v1/chat/completions` |

## Ports

- **4000**: ccproxy (Anthropic API compatible)
- **8450**: oca_api (OpenAI API compatible)

## Configuration Files

- `~/.ccproxy/config.yaml` - LiteLLM model deployments
- `~/.ccproxy/ccproxy.yaml` - ccproxy hooks and rules

## Useful Commands

```bash
# Check ccproxy status
ccproxy status

# View ccproxy logs
ccproxy logs -f

# Stop ccproxy
ccproxy stop

# Restart ccproxy
ccproxy restart
```

## Notes

- The `oca/gpt-5.3-codex` model only supports the responses endpoint, not chat completions
- We use `oca/gpt-5.2` for opus/sonnet because it supports chat completions
- Environment variables are set in `~/.zshrc`:
  - `ANTHROPIC_BASE_URL=http://127.0.0.1:4000`
  - `ANTHROPIC_API_KEY=dummy-key`
