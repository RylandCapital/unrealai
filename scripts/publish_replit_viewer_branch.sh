#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIEWER_DIR="$ROOT_DIR/replit_viewer"
WORKTREE_DIR="$ROOT_DIR/.replit_viewer_publish"
BRANCH_NAME="${REPLIT_VIEWER_BRANCH:-replit-viewer}"
REMOTE_NAME="${REPLIT_VIEWER_REMOTE:-origin}"

"$ROOT_DIR/scripts/package_replit_viewer.sh"

if [[ -d "$WORKTREE_DIR" && ! -d "$WORKTREE_DIR/.git" ]]; then
  echo "Publish worktree path exists but is not a git worktree: $WORKTREE_DIR" >&2
  exit 1
fi

git -C "$ROOT_DIR" fetch "$REMOTE_NAME" "$BRANCH_NAME" >/dev/null 2>&1 || true

if [[ ! -d "$WORKTREE_DIR" ]]; then
  if git -C "$ROOT_DIR" show-ref --verify --quiet "refs/remotes/$REMOTE_NAME/$BRANCH_NAME"; then
    git -C "$ROOT_DIR" worktree add -B "$BRANCH_NAME" "$WORKTREE_DIR" "$REMOTE_NAME/$BRANCH_NAME"
  else
    git -C "$ROOT_DIR" worktree add --detach "$WORKTREE_DIR"
    git -C "$WORKTREE_DIR" checkout --orphan "$BRANCH_NAME"
    git -C "$WORKTREE_DIR" rm -rf . >/dev/null 2>&1 || true
  fi
fi

find "$WORKTREE_DIR" -mindepth 1 -maxdepth 1 ! -name ".git" -exec rm -rf {} +
cp -R "$VIEWER_DIR"/. "$WORKTREE_DIR"/
find "$WORKTREE_DIR" -type d -name "__pycache__" -prune -exec rm -rf {} +

git -C "$WORKTREE_DIR" add -A
if git -C "$WORKTREE_DIR" diff --cached --quiet; then
  echo "No Replit viewer changes to publish."
  exit 0
fi

as_of="$(ROOT_DIR="$ROOT_DIR" python3 - <<'PY'
import json
from pathlib import Path

path = Path(__import__("os").environ["ROOT_DIR"]) / "replit_viewer" / "dashboard_data" / "summary.json"
try:
    summary = json.loads(path.read_text())
    print(summary.get("as_of") or summary.get("report_start_date") or "latest")
except Exception:
    print("latest")
PY
)"

git -C "$WORKTREE_DIR" commit -m "Update Replit viewer data (${as_of})"
git -C "$WORKTREE_DIR" push "$REMOTE_NAME" "$BRANCH_NAME"

echo "Published Replit viewer branch: $REMOTE_NAME/$BRANCH_NAME"
