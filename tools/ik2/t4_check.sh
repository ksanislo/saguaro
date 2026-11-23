#!/usr/bin/env bash
# tools/ik2/t4_check.sh - saguaro-ik2's card check (tools/ik2/t4_check.py) with a Python that has the tokenizer's
# modules.  Uses the cards named and nothing else:   tools/ik2/t4_check.sh --gpus 2,3 [t4_check.py's options]
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
py="${SAGUARO_PYTHON:-}"
if [ -z "$py" ]; then
  for c in "$here/../../.venv/bin/python" python3; do
    if "$c" -c 'import regex' 2>/dev/null; then py="$c"; break; fi
  done
fi
[ -n "$py" ] || { echo "no Python with the 'regex' module (set SAGUARO_PYTHON)"; exit 2; }
exec "$py" -B "$here/t4_check.py" "$@"
