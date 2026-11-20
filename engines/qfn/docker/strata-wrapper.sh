#!/bin/sh
# `strata` inside the container: the Terminal, with the address and the API key of the server this container runs, so that
# `docker exec <name> strata monitor --once`, `strata status` and `strata connect opencode` work without typing either.
# A key or address you set yourself (SAGUARO_API_KEY, SAGUARO_URL, --url) always wins.
KEY_FILE="${SAGUARO_KEY_FILE:-/data/state/api-key}"
if [ -z "${SAGUARO_API_KEY:-}" ] && [ -r "$KEY_FILE" ]; then
  SAGUARO_API_KEY=$(cat "$KEY_FILE"); export SAGUARO_API_KEY
fi
SAGUARO_URL="${SAGUARO_URL:-http://127.0.0.1:${PORT:-8080}}"; export SAGUARO_URL
exec /opt/strata/terminal/bin/strata "$@"
