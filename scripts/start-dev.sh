#!/usr/bin/env bash
set -euo pipefail

SKIP_FASTAPI=0
SKIP_KG_SERVICE=0
SKIP_FRONTEND=0

for arg in "$@"; do
  case "$arg" in
    --skip-fastapi) SKIP_FASTAPI=1 ;;
    --skip-kg-service) SKIP_KG_SERVICE=1 ;;
    --skip-frontend) SKIP_FRONTEND=1 ;;
    *)
      echo "Unknown argument: $arg" >&2
      exit 1
      ;;
  esac
done

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="$PROJECT_ROOT/backend"
FRONTEND_DIR="$PROJECT_ROOT/frontend"
PROCESSED_DIR="$PROJECT_ROOT/data/datasets/openbg_img/processed"
PRODUCTION_MODELS_DIR="$PROJECT_ROOT/ml/artifacts/production_models"
LOG_DIR="$PROJECT_ROOT/.runtime_logs"

mkdir -p "$LOG_DIR"

echo "Starting MMKG dev services..."
echo "Project root: $PROJECT_ROOT"

if [[ ! -f "$PROCESSED_DIR/data.csv" || ! -f "$PROCESSED_DIR/metadata.json" ]]; then
  echo "Warning: KG processed data is missing. The KG page may be blank." >&2
  echo "Run: bash scripts/install.sh" >&2
fi

if [[ ! -d "$PRODUCTION_MODELS_DIR" ]]; then
  echo "Warning: production models directory is missing. Attribute Completion may run in metadata-only mode." >&2
  echo "Expected: $PRODUCTION_MODELS_DIR" >&2
fi

start_background() {
  local name="$1"
  local workdir="$2"
  local command="$3"
  local logfile="$LOG_DIR/$name.log"

  echo "Starting $name..."
  (
    cd "$workdir"
    eval "$command"
  ) >"$logfile" 2>&1 &
  echo "$!" >"$LOG_DIR/$name.pid"
  echo "  pid: $(cat "$LOG_DIR/$name.pid")"
  echo "  log: $logfile"
}

if [[ "$SKIP_FASTAPI" -eq 0 ]]; then
  start_background "fastapi-8000" "$BACKEND_DIR" "python3 -m uvicorn app.main:app --reload --port 8000"
fi

if [[ "$SKIP_KG_SERVICE" -eq 0 ]]; then
  start_background "kg-flask-5000" "$BACKEND_DIR" "python3 flask_app.py"
fi

if [[ "$SKIP_FRONTEND" -eq 0 ]]; then
  start_background "frontend-3000" "$FRONTEND_DIR" "npm run dev"
fi

echo
echo "Started requested services in the background."
echo "Frontend: http://localhost:3000"
echo "FastAPI:  http://127.0.0.1:8000"
echo "KG Flask: http://127.0.0.1:5000"
echo
echo "Logs are in: $LOG_DIR"
echo "Stop services with: bash scripts/stop-dev.sh"
