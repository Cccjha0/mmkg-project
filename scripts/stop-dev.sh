#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/.runtime_logs"

if [[ ! -d "$LOG_DIR" ]]; then
  echo "No runtime log directory found."
  exit 0
fi

for pidfile in "$LOG_DIR"/*.pid; do
  [[ -e "$pidfile" ]] || continue
  pid="$(cat "$pidfile")"
  name="$(basename "$pidfile" .pid)"
  if kill -0 "$pid" >/dev/null 2>&1; then
    echo "Stopping $name pid=$pid"
    kill "$pid" || true
  fi
  rm -f "$pidfile"
done

echo "Done."
