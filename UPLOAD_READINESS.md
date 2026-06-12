# Upload Readiness

Date: 2026-06-12

Workspace: `/home/codexlab/moe-run-anyway-project`

## Verdict

Ready to upload as a measurement-first MoE controllability probe project.

The repository is not claiming real expert paging yet. It is uploadable because a fresh reader can understand the project surface, run a dependency-free local readiness command, and see the exact live model gates that remain deferred.

## Current Commits

- `61f11a0 Finalize uploadable project surface`
- `b1dd127 Record git baseline status`
- `c6483c6 Initialize MoE run-anyway project`

This readiness file records the final state after `61f11a0` and is committed as the follow-up readiness record.

## Files Changed In Upload-Surface Commit

- Added root `README.md`.
- Added `scripts/check_project.py`.
- Updated `.gitignore` for future Codex prompt/final/launcher artifacts.
- Replaced stale absolute Windows paths in `memory-moe-mvp/README.md`.
- Replaced stale absolute Windows links in:
  - `memory-moe-mvp/docs/controller-architecture.md`
  - `memory-moe-mvp/docs/probe-observability-notes.md`
- Changed `memory-moe-mvp/run_forward_probe_demo.py` defaults to relative paths.
- Changed `memory-moe-mvp/run_mixtral_probe.ps1` defaults to relative Windows paths.
- Removed transient launcher/session files from the tracked project surface:
  - `catchup_prompt.md`
  - `model_tests_prompt.md`
  - `codex-catchup-final.md`
  - `codex-model-tests-final.md`
  - `run-moe-catchup.sh`
  - `run-moe-model-tests-pass.sh`

Untracked upload-pass launcher artifacts were also removed locally and are covered by `.gitignore`.

## Commands Run

```bash
git status --short --branch
git ls-files
python3 scripts/check_project.py
git commit -m "Finalize uploadable project surface"
git show --stat --oneline --name-status HEAD
git log --oneline --decorate -5
```

Observed readiness result before this status record:

- model target registry: valid, 5 targets, all required classes covered
- unit tests: `Ran 21 tests`, `OK`
- `py_compile`: passed for 16 Python files
- docs portability: passed for primary docs

The final clean-tree command for the committed readiness record is:

```bash
python3 scripts/check_project.py --require-clean
```

## Deferred Live-Test Gates

No package install, model download, Docker start, live model server, or GPU-heavy job was run in this pass.

Still deferred:

- user-started `llama-server` with `--metrics --slots --props --perf` and log capture
- `llama_runtime_probe.py` against `data/mixtral_probe_prompts.json`
- `llama_sidecar.py` around a running upstream backend
- hookable PyTorch MoE run with a user-provided local checkpoint and compatible environment
- any backend patch or `libllama` harness that emits semantic expert ids from stock `llama.cpp`
- any real expert paging, eviction, pinning, or preload actuator

The exact live command shapes are in `memory-moe-mvp/docs/model-target-test-plan.md`.

## Notes For Future Agents

Treat the project as a probe stack and replay/advisor harness. The source of truth for target coverage is `memory-moe-mvp/data/model_target_registry.json`, validated by `memory-moe-mvp/model_target_registry.py` and the root readiness script.

Historical context remains in `context-pack/` and the prior status files. Those files are useful context, but they should not override the current upload surface described by the root README and this readiness record.
