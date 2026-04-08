#!/usr/bin/env bash
set -euo pipefail

# Wrapper entrypoint that forwards to the Python validator.
# Usage:
#   ./validate-submission.sh <ping_url> [repo_dir]
# If repo_dir is omitted and ./triage_env/openenv.yaml exists,
# it defaults to ./triage_env for convenience.

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

if [ "$#" -eq 1 ] && [ -f "$SCRIPT_DIR/triage_env/openenv.yaml" ]; then
	exec "$PYTHON_BIN" "$SCRIPT_DIR/validation.py" "$1" "$SCRIPT_DIR/triage_env"
fi

exec "$PYTHON_BIN" "$SCRIPT_DIR/validation.py" "$@"
