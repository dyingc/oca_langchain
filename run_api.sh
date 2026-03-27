#!/usr/bin/env bash
source .venv/bin/activate

# Refresh tokens if >6h since last forced refresh; no-op otherwise.
python -c "from core.token_utils import force_refresh_token; t = force_refresh_token(force=False); print(f'Access token ready: {t[:20]}...')"

uvicorn api:app --host 127.0.0.1 --port 8450
