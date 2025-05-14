#!/bin/bash

# Load environment variables from .env
set -a
[ -f .env ] && source .env
set +a

# Set defaults if not set in .env
# Note: dentify your Chrome application path:
#  default path is for MacOS.
CHROME_PATH=${CHROME_PATH:-"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"}
CHROME_USER_DATA=${CHROME_USER_DATA:-"./custom_chrome_profile"}
CHROME_REMOTE_DEBUG_PORT=${CHROME_REMOTE_DEBUG_PORT:-9222}

"$CHROME_PATH" \
  --remote-debugging-port="$CHROME_REMOTE_DEBUG_PORT" \
  --user-data-dir="$CHROME_USER_DATA" \
  --no-first-run \
  --no-default-browser-check