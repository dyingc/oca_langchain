#!/usr/bin/env bash

# Exit if uvicorn is already running
if /usr/bin/pgrep -f "uvicorn api:app --host 127.0.0.1 --port 8450" > /dev/null; then
	  exit 0
fi

# Activate venv
source .venv/bin/activate

# Reset token
sed -i '.bak' "s#^OAUTH_ACCESS_TOKEN=.*#OAUTH_ACCESS_TOKEN=#g" .env

# Run API with timeout
/opt/homebrew/bin/timeout 10 uvicorn api:app --host 127.0.0.1 --port 18450
