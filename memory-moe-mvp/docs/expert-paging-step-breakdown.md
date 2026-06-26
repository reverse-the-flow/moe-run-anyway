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
| Capture a passive sidecar artifact bundle. | active | Mixtral sidecar smoke bundle captured; expected bundle shape remains sidecar `manifest.json`, `events.jsonl`, `summary.json`, upstream observability snapshots when available, and Model Plane run metadata. | Needs repeated sidecar traffic for each selected backend if comparing sidecar behavior. |
| Capture a llama.cpp baseline artifact bundle. | active | Mixtral and Qwen3 GGUF baseline bundles captured; Nemotron Super GGUF failed at llama.cpp load due tensor-shape compatibility. | Needs no new prompt traffic for Nemotron Super until runtime compatibility is inspected. |
| Capture a vLLM or OpenAI-compatible baseline artifact bundle. | active | Nemotron Omni NVFP4 vLLM readiness and one-prompt baselines captured, including a 512-token-cap rerun with `finish_reason=stop`. | Needs repeated prompt-family coverage if comparing behavior, not just runtime compatibility. |
| Compare backend observability fields. | active | `docs/calliope-moe-architecture-pass.md` records first backend differences. | Needs a compact comparison report across Mixtral, Qwen3, and Nemotron Omni artifacts. |
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
| Classify current PC/GX10 hook candidates. | done | `data/hookable_moe_attempt_matrix.json` records PC HF non-MoEs, PC/GX10 GGUF engine-hook candidates, and the GX10 Nemotron HF dependency blocker. | None. |
| Select a hookable MoE target. | done | GX10 Mixtral/Qwen3 and PC Mixtral/Nemotron Cascade/Qwen3 Coder produced llama.cpp engine-hook traces; GX10 Nemotron HF remains a Python hook candidate blocked by missing trusted-code dependencies. | Python track still needs a compatible Transformers MoE runtime. |
| Build a thin Python target runner around `ForwardHookMoEProbe`. | planned | Expected: runner loads one user-provided local model, attaches hooks, runs prompt cases, writes existing probe artifacts, and does not create a new artifact shape. | Needs compatible Transformers target/runtime. |
| Build a minimal llama.cpp router trace patch. | done | `patches/llama-cpp-moe-router-trace-example.patch` adds an eval-callback trace example; GX10 and PC runs emitted selected expert ids and weights. | None for direct GGUF hook smoke. |
| Capture router outputs with hooks. | active | GX10 Mixtral: 96 validated events across 32 layers. GX10 Qwen3: 144 validated events across 48 layers. PC Mixtral: 96 events across 32 layers. PC Nemotron Cascade: 69 events across 23 routed layers. PC Qwen3 Coder: 144 events across 48 layers. | Needs Python-track target and repeatable launch-card packaging. |
| Preflight oversized GGUF MoE targets. | done | PC Llama 4 Scout metadata preflight found 48 routed MoE layers, 16 experts, top-k 1. PC DeepSeek V3.1 metadata preflight found 58 routed MoE layers, 256 experts, 1 shared expert, top-k 8. | Full router traces need a higher-memory host or Docker memory increase. |
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

1. Python hookable semantic routing still needs a compatible local/cloud
   Transformers MoE runtime.
2. Stock endpoints do not expose semantic expert routing.
3. PC Ollama itself is not instrumented with the hook yet, but local Ollama
   blobs are readable through direct patched llama.cpp.
4. Nemotron-H GGUF loading is blocked on current llama.cpp tensor-layout
   compatibility; GX10 still has the 86.8 GB local GGUF.
5. PC Llama 4 Scout and DeepSeek V3.1 full router traces need a higher-memory
   host or Docker memory increase.
6. No live runtime actuator exists.
7. Residency observation and residency control are missing.
8. Cleanup/restore proof is missing.
9. Processed expert-store work needs real checkpoint files and disk, but should
   not require loading the full model into RAM.

## Next Achievable Steps

1. Run the synthetic hook smoke after hookable-probe edits.
2. Package the llama.cpp Docker image and run command as a repeatable launch-card path.
3. Build the thinnest real-model runner around `ForwardHookMoEProbe` when a
   compatible Transformers MoE runtime is available.
4. Add the expert inventory manifest schema as a supporting offline track.
5. Produce the backend observability comparison from captured artifacts.
