#!/usr/bin/env bash

# Activate venv
source .venv/bin/activate

# Reset token
sed -i '.bak' "s#^OAUTH_ACCESS_TOKEN=.*#OAUTH_ACCESS_TOKEN=#g" .env

# Run API with timeout
/opt/homebrew/bin/timeout 10 uvicorn api:app --host 127.0.0.1 --port 18450
