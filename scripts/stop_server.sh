#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

if [[ -f devserver.pid ]]; then
  PID=$(cat devserver.pid || true)
  if [[ -n "${PID:-}" ]]; then
    if ps -p "$PID" >/dev/null 2>&1; then
      echo "[INFO] Stopping server PID $PID"
      kill -TERM "$PID" || true
      sleep 1
      if ps -p "$PID" >/dev/null 2>&1; then
        echo "[WARN] Force killing PID $PID"
        kill -9 "$PID" || true
      fi
    else
      echo "[INFO] No running process with PID $PID"
    fi
  fi
  rm -f devserver.pid
else
  echo "[INFO] No devserver.pid file, nothing to stop."
fi
