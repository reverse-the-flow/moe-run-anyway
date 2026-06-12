#!/usr/bin/env bash
set -euo pipefail

export PATH="/home/codexlab/.nvm/versions/node/v22.22.2/bin:$PATH"

WORKSPACE="/home/codexlab/moe-run-anyway-project"
PROMPT_FILE="$WORKSPACE/model_tests_prompt.md"
LOG_FILE="$WORKSPACE/codex-model-tests.log"

cd "$WORKSPACE"

{
  printf '=== Codex model-tests pass started at %s ===\n' "$(date --iso-8601=seconds)"
  codex exec \
    --cd "$WORKSPACE" \
    --sandbox danger-full-access \
    --output-last-message "$WORKSPACE/codex-model-tests-final.md" \
    - <"$PROMPT_FILE"
  printf '=== Codex model-tests pass finished at %s ===\n' "$(date --iso-8601=seconds)"
} >>"$LOG_FILE" 2>&1
