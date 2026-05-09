#!/usr/bin/env bash
set -euo pipefail

SKIP_PYTHON=0
SKIP_FRONTEND=0
SKIP_KG_DATA=0

for arg in "$@"; do
  case "$arg" in
    --skip-python) SKIP_PYTHON=1 ;;
    --skip-frontend) SKIP_FRONTEND=1 ;;
    --skip-kg-data) SKIP_KG_DATA=1 ;;
    *)
      echo "Unknown argument: $arg" >&2
      exit 1
      ;;
  esac
done

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRONTEND_DIR="$PROJECT_ROOT/frontend"
KG_DIR="$PROJECT_ROOT/kg"
PROCESSED_DIR="$PROJECT_ROOT/data/datasets/openbg_img/processed"
RAW_DIR="$PROJECT_ROOT/data/datasets/openbg_img/raw"

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Required command not found: $1" >&2
    exit 1
  fi
}

echo "MMKG setup starting..."
echo "Project root: $PROJECT_ROOT"

require_command python3
require_command npm

if [[ "$SKIP_PYTHON" -eq 0 ]]; then
  echo
  echo "[1/3] Installing Python dependencies..."
  (cd "$PROJECT_ROOT" && python3 -m pip install -r requirements.txt)
fi

if [[ "$SKIP_FRONTEND" -eq 0 ]]; then
  echo
  echo "[2/3] Installing frontend dependencies..."
  (cd "$FRONTEND_DIR" && npm install)
fi

if [[ "$SKIP_KG_DATA" -eq 0 ]]; then
  echo
  echo "[3/3] Checking KG processed data..."
  DATA_CSV="$PROCESSED_DIR/data.csv"
  METADATA_JSON="$PROCESSED_DIR/metadata.json"
  TRAIN_TSV="$RAW_DIR/OpenBG-IMG_train.tsv"

  if [[ ! -f "$TRAIN_TSV" ]]; then
    echo "Warning: raw OpenBG-IMG files were not found. Skipping KG data generation." >&2
    echo "Expected: $TRAIN_TSV" >&2
  else
    mkdir -p "$PROCESSED_DIR"

    if [[ ! -f "$DATA_CSV" ]]; then
      echo "Generating data.csv..."
      (cd "$KG_DIR" && python3 convert_openbg.py)
    else
      echo "data.csv already exists."
    fi

    if [[ ! -f "$METADATA_JSON" ]]; then
      echo "Generating metadata.json..."
      (cd "$KG_DIR" && python3 generate_metadata.py)
    else
      echo "metadata.json already exists."
    fi
  fi
fi

echo
echo "Setup complete."
echo "Run the app with:"
echo "  bash scripts/start-dev.sh"
