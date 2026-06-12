You are continuing work on the MoE Run Anyway / memory-moe-mvp project.

Current date: 2026-06-12.

Remote workspace:
- /home/codexlab/moe-run-anyway-project
- Source/prototype snapshot: memory-moe-mvp/
- Source context notes: context-pack/
- Prior catch-up report: CATCHUP_STATUS.md

Scope:
1. Treat /home/codexlab/moe-run-anyway-project as the only writable project workspace.
2. Make the workspace a useful git project if it is not already:
   - verify git status and current branch
   - create or update a sensible .gitignore for Python caches, logs, temp outputs, generated run dirs, and Codex session logs
   - preserve source files, docs, tests, data fixtures, Docker files, context notes, and catch-up/status docs
   - create an initial baseline commit if the repository has no commits and git identity allows it
   - do not rewrite history or delete copied source/context artifacts
3. Work toward tests for different MoE model targets.
   - Ground this in the existing probes and contracts, not a fresh architecture rewrite.
   - Add durable repo artifacts such as a model-target matrix, target registry/config fixture, test-tier plan, or lightweight validation harness if useful.
   - Cover at least these target classes:
     - stock llama.cpp/OpenAI-compatible server path where semantic expert ids may not be exposed
     - hookable PyTorch-style MoE runtime where router/expert activations can be observed
     - passive sidecar/proxy observation path
     - small locally cached or easily provided MoE targets such as OLMoE-style models, if the repo can support them without downloads
     - Mixtral-style targets already referenced by the project
   - Make the next real live model test command(s) explicit, but do not run live model servers, Docker, downloads, or GPU-heavy jobs.
4. Run only light local validation:
   - unit tests if available
   - py_compile for touched Python
   - any new pure local validation script you add
5. Produce /home/codexlab/moe-run-anyway-project/MODEL_TESTS_STATUS.md with:
   - git state and commit hash if committed
   - files changed
   - what model-test direction was added
   - commands/checks run
   - exact blockers or deferred live-test requirements

Constraints:
- Do not install packages.
- Do not download models.
- Do not start Docker.
- Do not launch a real llama-server or model server.
- Do not start GPU-heavy jobs.
- Keep edits focused and small.
- If you cannot commit, leave the repository initialized with a clear status explaining why.

Goal:
Make the project easier to resume as a real git repo and move it toward a disciplined suite of tests across multiple MoE model/runtime families.
