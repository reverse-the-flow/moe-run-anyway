# Expert Paging Step Breakdown

This breaks the roadmap into small steps so blockers are visible early. A step
should not advance unless its evidence is written as an artifact, fixture,
manifest, report, or test output.

## Status Labels

- `done`: implemented and validated in the repository.
- `active`: current work can proceed without changing the roadmap.
- `blocked`: the next action needs a model, runtime, machine, credential, or
  upstream patch that is not currently available.
- `planned`: valid next work, but another step should land first.

## Phase 0: Harness Contract And Model Plane Handoff

| Step | Status | Evidence | Blocker |
| --- | --- | --- | --- |
| Define the Model Plane manifest handoff. | done | Saved manifest fixtures and manifest planner. | None. |
| Define managed expert loading vocabulary. | done | Managed loading contract and machine-readable plan. | None. |
| Add no-side-effect validation commands. | done | Planner scripts validate without downloads, secrets, Docker, servers, or prompt traffic. | None. |
| Separate runtime evidence from semantic routing evidence. | done | Roadmap truth table and artifact contracts. | None. |

## Phase 1: Runtime Baseline Evidence

| Step | Status | Evidence | Blocker |
| --- | --- | --- | --- |
| Validate runtime baseline artifact shape. | active | `scripts/plan_runtime_baseline_artifacts.py`. | None. |
| Confirm passive sidecar implementation. | done | `llama_sidecar.py` forwards traffic, captures request/response summaries, optional upstream observability, and shared-contract artifacts. | None. |
| Capture a passive sidecar artifact bundle. | blocked | Expected: sidecar `manifest.json`, `events.jsonl`, `summary.json`, upstream observability snapshots when available, and Model Plane run metadata. | Needs an approved running upstream endpoint and traffic routed through the sidecar. |
| Capture a llama.cpp baseline artifact bundle. | blocked | Expected: metrics, slots, props, timings, log-growth summary, Model Plane manifest. | Needs a user-approved running `llama-server` with observability enabled. |
| Capture a vLLM or OpenAI-compatible baseline artifact bundle. | blocked | Expected: readiness, model metadata, timing, endpoint behavior, run manifest. | Needs an approved running endpoint. |
| Compare backend observability fields. | planned | Expected: report labeling what each backend can and cannot prove. | Needs at least one artifact bundle per selected backend. |
| Produce a Phase 1 blocker report. | planned | Expected: missing capability list for inventory, routing visibility, residency read/write, fallback, cleanup, and artifact export. | Needs baseline artifacts. |

## Phase 1B: Offline Expert Store Preparation

This track is added because live paging needs a processed expert store, but that
store should not require loading the full model into RAM.

| Step | Status | Evidence | Blocker |
| --- | --- | --- | --- |
| Define an expert inventory manifest schema. | active | Expected: layer id, expert id, tensor names, source shard, byte offsets or tensor metadata, dtype, shape, estimated bytes. | Needs target model format examples. |
| Build a safetensors shard scanner. | planned | Expected: scanner can read shard indexes and tensor metadata without loading tensor values. | Needs local or cloud-accessible checkpoint files. |
| Build a streaming expert packer dry run. | planned | Expected: dry run reports output layout and required disk without writing tensors. | Needs scanner output. |
| Build a streaming expert packer write path. | planned | Expected: writes packed expert files and manifest by streaming tensor data. | Needs enough disk and a target checkpoint. |
| Validate partial-shard behavior. | planned | Expected: partial inventory report says exactly which experts/tensors are missing. | Needs intentionally incomplete shard set. |
| Validate full-store consistency. | blocked | Expected: sampled tensor hashes or byte ranges match source checkpoint. | Needs complete checkpoint files, but not full RAM load. |

Partial shards are useful for inventory and failure-mode testing. They are not
valid evidence for learned routing policy.

## Phase 2: Semantic Routing Traces

This is the current active focus. Passive sidecar evidence stays useful, but
semantic routing should come from hookable internals before any fork,
controller, or live actuator work.

| Step | Status | Evidence | Blocker |
| --- | --- | --- | --- |
| Confirm synthetic hook smoke path. | done | `run_forward_probe_demo.py` exercises hook attach, router event capture, spans, and window summaries without a model. | None. |
| Select a hookable MoE target. | active | Expected: target note with model family, runtime, memory needs, and hook points. | Needs a local small MoE, a cloud box, or a user-supplied checkpoint/runtime. |
| Build a thin target runner around `ForwardHookMoEProbe`. | active | `run_transformers_forward_probe.py --dry-run` validates a user-provided local Transformers model path, reports missing deps, refuses token-env/download paths, and emits the guarded run command. | Needs selected target to execute without `--dry-run`. |
| Capture router outputs with hooks. | planned | Expected: layer id, expert ids, scores/probabilities, entropy, token/window metadata. | Needs selected hookable runtime. |
| Capture dense or full-runtime fallback output. | planned | Expected: baseline outputs for the same prompt set. | Needs runnable target. |
| Validate trace artifacts against the shared contract. | planned | Expected: trace validation report. | Needs trace artifacts. |
| Record hook failure modes. | planned | Expected: failures are explicit and do not downgrade to timing inference. | Needs hook attempts. |

If local hardware cannot run the target, cloud is the clean route for this
phase. Running with fewer shards can test error handling, but it cannot produce
trustworthy routing traces.

## Phase 3: Replay Policy Audit

| Step | Status | Evidence | Blocker |
| --- | --- | --- | --- |
| Define baseline policies. | planned | Expected: observe-only, keep-hot, preload-shortlist, evict-cold, fallback-dense policy names. | Needs Phase 2 trace shape. |
| Replay policies over semantic traces. | planned | Expected: warm-hit rate, miss rate, churn, fallback frequency, candidate-set size. | Needs semantic traces. |
| Compare against dense or full-runtime fallback. | planned | Expected: quality or behavior delta report. | Needs fallback outputs. |
| Select one candidate policy for live spike. | planned | Expected: policy choice with rejection reasons for alternatives. | Needs replay report. |

## Phase 4: Backend Adapter Feasibility Spike

| Step | Status | Evidence | Blocker |
| --- | --- | --- | --- |
| Choose first backend substrate. | planned | Expected: comparison of llama.cpp patch, vLLM extension, hookable PyTorch adapter, MoE-Infinity-style adapter, and prototype runtime. | Needs Phase 1 and Phase 3 reports. |
| Identify routing visibility control point. | blocked | Expected: exact function, hook, API, or patch point that exposes routed expert ids before expert load. | Needs backend source inspection and/or fork spike. |
| Identify residency control point. | blocked | Expected: exact place to preload, pin, evict, demote, or load expert tensors. | Needs backend source inspection and/or fork spike. |
| Define cleanup and rollback. | planned | Expected: restore dense/full-residency behavior or prove no mutation occurred. | Needs selected control point. |
| Emit live-adapter artifact shape. | planned | Expected: before/after residency, policy decision, fallback, cleanup, and timing artifacts. | Needs selected adapter design. |

## Phase 5: Guarded Live Actuator Experiment

| Step | Status | Evidence | Blocker |
| --- | --- | --- | --- |
| Implement actuator behind a kill switch. | blocked | Expected: live mode can be disabled instantly. | Needs Phase 4 control point. |
| Run a tiny smoke model or tiny MoE first. | planned | Expected: successful preload/evict/pin/fallback cycle with cleanup. | Needs actuator implementation and test model. |
| Run one constrained real-model experiment. | planned | Expected: memory or latency effect compared with baseline. | Needs smoke success and approved hardware. |
| Compare against replay expectations. | planned | Expected: live result either matches replay assumptions or explains mismatch. | Needs live artifact bundle. |
| Decide whether to continue, fork, or stop. | planned | Expected: go/no-go decision with maintenance cost. | Needs live evidence. |

## Current Critical Blockers

1. Hookable semantic routing needs a runnable target: local small MoE,
   user-provided checkpoint/runtime, or cloud run.
2. Stock endpoints do not expose semantic expert routing.
3. No live runtime actuator exists.
4. Residency observation and residency control are missing.
5. Cleanup/restore proof is missing.
6. Processed expert-store work needs real checkpoint files and disk, but should
   not require loading the full model into RAM.

## Next Achievable Steps

1. Run the synthetic hook smoke after hookable-probe edits.
2. Select one small hookable MoE target for semantic routing traces.
3. Dry-run `run_transformers_forward_probe.py` against the selected local
   Transformers model directory, then run it without `--dry-run` once the local
   path and optional dependencies are ready.
4. Capture one approved runtime baseline from a running backend.
5. Add the expert inventory manifest schema as a supporting offline track.
