#!/usr/bin/env bash
set -euo pipefail
# Basic health check for the app. Intended to be run by cron every 5 minutes.
# It reads the devserver.port if present, otherwise defaults to 8000.

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PORT_FILE="$BASE_DIR/devserver.port"
PORT="8000"
if [[ -f "$PORT_FILE" ]]; then PORT="$(cat "$PORT_FILE" | tr -d '\n' | tr -d '\r')"; fi

URL="http://127.0.0.1:${PORT}"

ok=1

# Check root page
if ! curl -fsS "$URL/" >/dev/null 2>&1; then
  echo "[FAIL] UI not responding at $URL/"
  ok=0
else
  echo "[OK] UI responding"
fi

# Check health endpoint
if ! curl -fsS "$URL/api/health" >/dev/null 2>&1; then
  echo "[FAIL] /api/health not responding"
  ok=0
else
  echo "[OK] /api/health responding"
fi

# Optional: Check Gemini status from /api/health
GEMINI_ENABLED=$(curl -fsS "$URL/api/health" | grep -o '"enabled":\s*true' || true)
if [[ -z "$GEMINI_ENABLED" ]]; then
  echo "[WARN] Gemini not enabled (set GEMINI_API_KEY in .env)"
else
  echo "[OK] Gemini enabled"
fi

if [[ $ok -eq 0 ]]; then
  exit 1
fi

exit 0
