#!/usr/bin/env bash
set -euo pipefail

# Wrapper entrypoint that forwards to the Python validator.
# Usage:
#   ./validate-submission.sh <ping_url> [repo_dir]
# If repo_dir is omitted, validation.py defaults to the repository root.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-}"

if [ -z "$PYTHON_BIN" ]; then
	if [ -x "$SCRIPT_DIR/.venv/bin/python" ]; then
		PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
	elif command -v python3 >/dev/null 2>&1; then
		PYTHON_BIN="python3"
	elif command -v python >/dev/null 2>&1; then
		PYTHON_BIN="python"
	else
		echo "Error: no Python interpreter found. Set PYTHON_BIN or create .venv." >&2
		exit 127
	fi
fi

exec "$PYTHON_BIN" "$SCRIPT_DIR/validation.py" "$@"
