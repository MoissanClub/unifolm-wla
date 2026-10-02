#!/usr/bin/env bash
# Full existing structured task for comparison. This is NOT a UniFoLM policy rollout.
set -euo pipefail
workspace=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$workspace/g1-fridge-fetch"
case "${1:-sim}" in
  sim) exec "${PYTHON_BIN:-python3}" -m g1_fetch.cli run --dry-run --sim ;;
  check) exec "${PYTHON_BIN:-python3}" -m g1_fetch.cli check --config configs/pc2.yaml ;;
  *) echo "Use sim or check. See README for staged physical baseline evaluation." >&2; exit 2 ;;
esac
