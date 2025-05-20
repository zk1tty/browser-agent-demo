#!/bin/bash

# Load environment variables from .env
set -a
[ -f .env ] && source .env # .env might set CHROME_REMOTE_DEBUG_PORT or CHROME_CDP_URL
set +a

# Port for Chrome to listen on for remote debugging
# User can set CHROME_REMOTE_DEBUG_PORT in .env to override default
CHROME_REMOTE_DEBUG_PORT=${CHROME_REMOTE_DEBUG_PORT:-9222}

# Set defaults if not set in .env for other parameters
CHROME_PATH=${CHROME_PATH:- "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"}
CHROME_USER_DATA=${CHROME_USER_DATA:- "./custom_chrome_profile"}

echo "Starting Chrome with remote debugging on port: $CHROME_REMOTE_DEBUG_PORT"
echo "User data directory: $CHROME_USER_DATA"
echo "Ensure your .env or environment sets CHROME_CDP_URL=http://localhost:$CHROME_REMOTE_DEBUG_PORT (or the correct host/port if different) for webui.py to connect."

"$CHROME_PATH" \
  --remote-debugging-port="$CHROME_REMOTE_DEBUG_PORT" \
  --user-data-dir="$CHROME_USER_DATA" \
  --no-first-run \
  --no-default-browser-check