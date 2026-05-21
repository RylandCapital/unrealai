#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIEWER_DIR="$ROOT_DIR/replit_viewer"
SOURCE_DASHBOARD="$ROOT_DIR/unrealai/dashboard.py"
SOURCE_DATA="$ROOT_DIR/unrealai/dashboard_data"
SOURCE_ALLOCATIONS="$ROOT_DIR/unrealai/allocations"
OUT_DIR="$ROOT_DIR/dist"
ZIP_PATH="$OUT_DIR/replit_viewer.zip"

if [[ ! -f "$SOURCE_DASHBOARD" ]]; then
  echo "Missing dashboard source: $SOURCE_DASHBOARD" >&2
  exit 1
fi

if [[ ! -d "$SOURCE_DATA" ]]; then
  echo "Missing dashboard data folder: $SOURCE_DATA" >&2
  echo "Run live_report.py first so dashboard_data exists." >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
cp "$SOURCE_DASHBOARD" "$VIEWER_DIR/dashboard.py"
rm -rf "$VIEWER_DIR/dashboard_data"
cp -R "$SOURCE_DATA" "$VIEWER_DIR/dashboard_data"
if [[ -d "$SOURCE_ALLOCATIONS" ]]; then
  python3 "$ROOT_DIR/scripts/generate_model_symbols.py" \
    "$SOURCE_ALLOCATIONS" \
    "$VIEWER_DIR/dashboard_data/model_symbols.json" \
    "$VIEWER_DIR/dashboard_data/model_symbols.csv"
else
  echo "[warn] allocation folder not found, model filters will use available local workbooks only: $SOURCE_ALLOCATIONS" >&2
fi

find "$VIEWER_DIR" -type d -name "__pycache__" -prune -exec rm -rf {} +
rm -f "$ZIP_PATH"

(
  cd "$VIEWER_DIR"
  zip -qr "$ZIP_PATH" .
)

echo "Packaged Replit viewer: $ZIP_PATH"
