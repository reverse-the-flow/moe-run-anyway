# MoE Run Anyway / memory-moe-mvp Catch-Up Status

Date: 2026-06-12

Workspace read:

- `/home/codexlab/moe-run-anyway-project/memory-moe-mvp`
- `/home/codexlab/moe-run-anyway-project/context-pack`

This is a catch-up note, not an implementation pass. I did not install packages, download models, contact external network services, start a real `llama-server`, start Docker, or run GPU-heavy jobs.

## Current Architecture

The project has shifted from "MVP that runs too-big MoEs" toward a measurement-first MoE controllability probe with a small simulator/controller replay path.

Main moving parts:

- `moe_shared_contract.py`: shared bridge schema, currently `memory-moe-bridge-v1`, with required fields such as `probe_tier`, `backend_family`, `prompt_family`, `window_size_tokens`, `policy_name`, `resident_budget_fraction`, `candidate_set_size`, `fallback_used`, `warm_hit_rate`, `miss_rate`, `churn`, `eviction_regret`, `context_retention`, and `dense_baseline_delta`.
- `memory_moe.py`: synthetic logical-residency simulator. It models fixed core memory, dynamic expert residency, LRU eviction, optional context-preserving eviction, load latency, context retention, and synthetic quality proxy.
- `llama_sidecar.py`: passive external sidecar/proxy for OpenAI-compatible or `llama-server` requests. It logs request/response summaries, latency, system memory, optional GPU snapshots, and optional upstream `/metrics`, `/slots`, and `/props` snapshots.
- `llama_runtime_probe.py`: active `llama.cpp` runtime probe. It submits prompt-suite requests directly to `llama-server`, captures before/after `/metrics`, `/slots`, `/props`, optional log-file growth, response usage/timings, and writes correlated artifacts.
- `moe_forward_probe.py`: Python forward-hook probe for hookable PyTorch-style MoE runtimes. It does not import torch at module import time. It attaches to modules with router-like names, extracts explicit expert ids/weights or router logits from outputs, and writes per-router events plus window summaries.
- `run_forward_probe_demo.py`: synthetic hook driver using fake router modules. It exists to generate forward-probe artifacts without a real PyTorch MoE backend.
- `moe_controller_demo.py`: replay/advisor/controller demo. It ingests synthetic traces or forward-probe window summaries, runs dense baseline plus budget sweeps at 40% and 25%, compares `reactive_lru`, `weighted_lru`, and `window_predictive`, produces an advisor report, failure-mode matrix, and next-stage branch recommendation.
- `data/mixtral_probe_prompts.json`: deterministic-ish prompt corpus for `dolphin-mixtral:8x7b`, spanning prose, code, SQL, structured data, math, logs, shell, multilingual, and hybrid prompts.
- `data/synthetic_controller_trace.json`: synthetic layer/window trace for the controller replay.
- `data/toy_workload.json`: synthetic simulator workload for the original memory/context tradeoff.
- `docker/`: separate images/materials for llama sidecar and forward-hook probe. The forward-hook compose file currently starts an interactive/sleeping container, not a fixed test runner.

The intended tiering is:

- Tier 1: passive external observation, currently `llama_sidecar.py`.
- Tier 2A: stock `llama.cpp` internal-runtime observation through exposed server surfaces, currently `llama_runtime_probe.py`.
- Tier 2B: Python semantic-routing observation through forward hooks, currently `moe_forward_probe.py`.
- Advisor/replay: offline policy scoring and budget sweeps, currently `moe_controller_demo.py`.
- Controller: only demo/replay level right now; no live runtime intervention is implemented.

## What Appears To Work

Local lightweight verification passed.

Observed working behavior:

- All local unit tests pass under Python 3.12.3.
- Python syntax compilation of key modules succeeds.
- The toy simulator loads `data/toy_workload.json`, prints the plan, and writes `manifest.json`, `events.jsonl`, and `summary.json`.
- The toy simulator demonstrates the expected tradeoff:
  - `context_reserve`: higher quality/context, more load/eviction work.
  - `reuse_bias`: lower latency than baseline, similar context retention.
  - `baseline`: simple top-k plus LRU.
- The forward-hook synthetic demo writes real probe artifacts into `/tmp`, including router events and window summaries.
- The replay/controller demo runs against `data/synthetic_controller_trace.json` and writes dense baseline, budget sweeps, controller demo, advisor report, failure-mode matrix, and next-stage branch.
- The stubbed runtime-probe tests exercise a local in-process HTTP stub, not a real model server, and verify `/metrics`, `/slots`, `/props`, request/response handling, metric deltas, and log-growth capture.
- Shared contract wiring is covered across simulator, forward probe, sidecar accumulator, and runtime accumulator.

Specific observed smoke outputs:

- `python3 -m unittest discover -s tests`
  - `Ran 18 tests in 0.565s`
  - `OK`
- `python3 memory_moe.py run --output-dir /tmp/memory-moe-catchup-sim-runs`
  - `context_reserve: quality=0.969, latency_ms=101.600, context=0.948, loads=14, evictions=12`
  - `baseline: quality=0.802, latency_ms=68.667, context=0.458, loads=9, evictions=4`
  - `reuse_bias: quality=0.795, latency_ms=65.417, context=0.462, loads=8, evictions=3`
- `python3 moe_controller_demo.py --output-dir /tmp/memory-moe-catchup-controller-runs --label catchup-smoke`
  - Wrote `/tmp/memory-moe-catchup-controller-runs/20260612-164612-catchup-smoke/summary.json`
  - `bridge_status.status = paused`
  - advisor overall: `family_count=5`, `compact_family_count=4`, `chaotic_family_count=1`, `viability_assessment=predictive_residency_viable`
  - controller demo: `mean_dense_baseline_delta=0.001`, `mean_miss_rate=0.146`, `mean_churn=0.484`, `fallback_rate=0.5`
  - next-stage branch: `yellow`, "Stay at the advisor/controller layer and tighten traces before deeper descent."
- `python3 run_forward_probe_demo.py --output-dir /tmp/memory-moe-catchup-forward-runs --suite-path /home/codexlab/moe-run-anyway-project/memory-moe-mvp/data/mixtral_probe_prompts.json --max-prompts 2 --window-size-events 2 --label catchup-forward-smoke`
  - Wrote `/tmp/memory-moe-catchup-forward-runs/20260612-164612-catchup-forward-smoke`
  - summary totals: `router_event_count=4`, `token_count=12`, `unique_experts_seen=5`, `mean_routing_entropy=0.783`

## Known Gaps And Risks

- No real Mixtral, Kimi, GLM, or other live MoE backend was run during this catch-up pass.
- No real Week 1 shared baseline report over a live Mixtral/`llama.cpp` path appears to be present. The repo has schema and probes, but not the actual current-backend baseline evidence artifact.
- The forward-hook bridge is still synthetic/demo unless a hookable PyTorch MoE backend is wired in. `moe_controller_demo.build_bridge_status()` explicitly marks synthetic/demo traces as `paused`.
- The `llama.cpp` probes do not expose semantic expert ids. They can correlate request timing, metrics, slots, props, and logs, but not per-layer routed experts.
- The controller path is replay/logical only. It does not load, evict, pin, or preload real experts in a live runtime.
- The simulator and controller quality metrics are synthetic proxies. Useful for plumbing and relative policy behavior, not real model quality.
- `context_retention_for_fraction()` in `moe_controller_demo.py` is a simplified heuristic, not actual KV-cache accounting.
- `run_forward_probe_demo.py` has Windows `X:/Experiments/...` defaults. It works on this Linux workspace only when `--output-dir` and `--suite-path` are supplied explicitly.
- README and docs still contain many Windows `X:/...` absolute links, which are stale in this Linux workspace.
- Docker forward-hook compose currently uses `sleep infinity`; it is useful as a workbench but not yet a fixed, approval-friendly test/demo workflow.
- External model/backend claims from the April notes were not reverified because this task explicitly asked not to contact network services.
- The current parent directory is not a Git repository, so I could not use git status to verify all workspace changes.

## Likely Next Actions

1. Freeze the shared baseline contract in one concrete command/script for the current `dolphin-mixtral:8x7b` path.
2. Run a real `llama.cpp` observational baseline with `--metrics --slots --props --perf --log-file ...` enabled, using `data/mixtral_probe_prompts.json` and at least two repeats for early prompt families.
3. Write the first real baseline report artifact that joins sidecar/runtime probe results into the shared contract.
4. Wire one real hookable PyTorch MoE target into `moe_forward_probe.py`, or explicitly mark the forward-hook bridge paused until model/runtime selection is resolved.
5. Convert Docker compose into fixed-command services for safe local verification:
   - unit tests
   - forward-hook synthetic demo
   - controller replay demo
   - optional runtime probe against a user-supplied running server
6. Add a small loader-boundary inspection checklist/script for any candidate backend:
   - parameter names and shapes
   - expert-specific names after load
   - packed/fused tensors
   - expert id to slice/offset maps
   - offload/paging abstractions
7. Keep the next implementation pass above real paging unless the real baseline and hook/probe data show stable working sets and enough backend controllability.

## Commands And Checks Run

Read/inspection commands:

```bash
sed -n '1,240p' README.md
sed -n '241,520p' README.md
rg --files docs
find . -maxdepth 2 -type f | sort
sed -n '1,260p' docs/probe-observability-notes.md
sed -n '1,260p' docs/controller-architecture.md
sed -n '1,520p' docs/thread-export-2026-04-24.md
wc -l docs/*.md README.md
rg --files /home/codexlab/moe-run-anyway-project/context-pack
wc -l *.md
sed -n '1,1147p' 'ChatGPT-Custom Kernel Exploration.md'
sed -n '1,2411p' 'ChatGPT-Model Loading and Quantization.md'
rg -n "^(#|##|[0-9]+\\.|Stage|Phase|First|Second|Third|Fourth|Where this|What you|Metrics|The first|Bottom line|Recommendation|Target|Concrete|Risks|Failure|Implementation|Architecture|Probe|Advisor|Controller)" context-pack/*.md
rg --files -g '*.py'
wc -l *.py tests/*.py
rg -n "^(class|def|@dataclass|PROFILES|DEFAULT|SHARED|CONTRACT|def main|if __name__)" *.py
sed -n '1,900p' memory_moe.py
sed -n '1,1320p' moe_controller_demo.py
sed -n '1,820p' moe_forward_probe.py
sed -n '1,620p' llama_runtime_probe.py
sed -n '1,1060p' llama_sidecar.py
sed -n '1,240p' run_forward_probe_demo.py
find docker -maxdepth 1 -type f -print -exec sed -n '1,220p' {} \;
find data -maxdepth 1 -type f -print -exec wc -l {} \;
sed -n '1,260p' tests/test_*.py
python3 --version
git status --short
```

Verification commands:

```bash
python3 -m unittest discover -s tests
python3 memory_moe.py plan
python3 moe_controller_demo.py --output-dir /tmp/memory-moe-catchup-controller-runs --label catchup-smoke
python3 run_forward_probe_demo.py \
  --output-dir /tmp/memory-moe-catchup-forward-runs \
  --suite-path /home/codexlab/moe-run-anyway-project/memory-moe-mvp/data/mixtral_probe_prompts.json \
  --max-prompts 2 \
  --window-size-events 2 \
  --label catchup-forward-smoke
python3 memory_moe.py run --output-dir /tmp/memory-moe-catchup-sim-runs
python3 -m py_compile memory_moe.py moe_shared_contract.py moe_forward_probe.py run_forward_probe_demo.py moe_controller_demo.py llama_runtime_probe.py llama_sidecar.py
```

No-network/no-model note:

- I did not run `llama_sidecar.py` as a live proxy.
- I did not run `llama_runtime_probe.py` against a real `llama-server`.
- I did not run Docker.
- I did not download model weights or Python packages.

## Blockers / Exact Errors

No blocker prevented producing this catch-up file.

The main concrete limitation is that this workspace root is not a Git repository:

```text
fatal: not a git repository (or any of the parent directories): .git
```

The main project blocker remains absence of real backend evidence in this local pass:

- no live `llama.cpp` Mixtral baseline was run
- no real hookable MoE backend was wired into the forward-hook probe
- no live loader/controller integration exists yet

## Bottom Line

The repo is in a useful measurement-first state. It has a shared trace contract, a passive sidecar, a stock `llama.cpp` runtime probe, a hookable-runtime forward probe, synthetic demos, a replay/advisor/controller demo, tests, and Docker scaffolding.

The next implementation pass should not jump straight to real expert paging. The grounded next move is to produce a real shared baseline report from a live MoE backend, then decide whether to continue with probe/advisor refinement, a hookable PyTorch bridge, or a loader-boundary descent.
