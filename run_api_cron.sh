#!/usr/bin/env bash
source .venv/bin/activate

# Always force a fresh token rotation on cron runs.
python -c "from core.token_utils import force_refresh_token; t = force_refresh_token(force=True); print(f'Access token refreshed: {t[:20]}...')"

/opt/homebrew/bin/timeout 10 uvicorn api:app --host 127.0.0.1 --port 18450
