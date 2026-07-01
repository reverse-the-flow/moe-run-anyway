# Memory-Aware MoE MVP

This package is a focused probe and replay harness for memory-aware Mixture-of-Experts runtime experiments. It treats the current problem as measurement and controllability first: learn what a backend exposes, score simple residency policies offline, and avoid claiming real expert paging before the runtime supports it.

## Current Scope

Implemented local surfaces:

- [memory_moe.py](memory_moe.py): synthetic logical-residency simulator.
- [moe_controller_demo.py](moe_controller_demo.py): replay/advisor/controller demo over synthetic traces or forward-probe window summaries.
- [llama_sidecar.py](llama_sidecar.py): passive sidecar/proxy around an OpenAI-compatible or `llama-server` upstream.
- [llama_runtime_probe.py](llama_runtime_probe.py): active stock `llama.cpp` runtime probe using `/metrics`, `/slots`, `/props`, response timings, and optional log growth.
- [moe_forward_probe.py](moe_forward_probe.py): forward-hook probe for hookable PyTorch-style MoE runtimes.
- [run_forward_probe_demo.py](run_forward_probe_demo.py): no-model synthetic hook driver.
- [run_transformers_forward_probe.py](run_transformers_forward_probe.py): guarded local-only Transformers runner for hookable MoE targets.
- [model_target_registry.py](model_target_registry.py): dependency-free validator for [data/model_target_registry.json](data/model_target_registry.json).

Not implemented yet:

- live expert loading, eviction, pinning, or paging
- semantic expert ids from stock `llama.cpp` or stock Ollama serving
- a bundled live model server, model weights, or package installer
- GPU-heavy or Docker-required validation in the local readiness path

## Shared Contract

Probe outputs are aligned around [moe_shared_contract.py](moe_shared_contract.py), currently `memory-moe-bridge-v1`. Important fields include:

- `probe_tier`
- `backend_family`
- `prompt_family`
- `window_size_tokens`
- `policy_name`
- `resident_budget_fraction`
- `candidate_set_size`
- `fallback_used`
- `warm_hit_rate`
- `miss_rate`
- `churn`
- `eviction_regret`
- `context_retention`
- `dense_baseline_delta`

The contract is meant to keep simulator, sidecar, runtime, hook, and controller artifacts comparable without pretending they expose the same depth of model internals.

## Probe Tiers

### Tier 0: Local Replay And Shape Checks

Run without a model server:

```bash
python3 memory_moe.py plan
python3 memory_moe.py run --output-dir sim-runs
python3 moe_controller_demo.py --output-dir controller-runs --label local-replay
python3 run_forward_probe_demo.py \
  --output-dir forward-probe-runs \
  --suite-path data/mixtral_probe_prompts.json \
  --max-prompts 2 \
  --window-size-events 2 \
  --label synthetic-hook-smoke
```

### Tier 1: Passive Sidecar

Use when a backend is already running and the goal is non-invasive request-boundary telemetry:

```bash
python3 llama_sidecar.py \
  --listen-port 8091 \
  --upstream-base-url http://127.0.0.1:18080 \
  --output-dir sidecar-runs \
  --label mixtral-sidecar \
  --capture-upstream-observability
```

Point the client at `http://127.0.0.1:8091` instead of the upstream server. This writes `manifest.json`, `events.jsonl`, and `summary.json`.

### Tier 2A: Stock llama.cpp Runtime Probe

Start `llama-server` yourself with observability enabled:

```bash
llama-server \
  --metrics \
  --slots \
  --props \
  --perf \
  --log-file /path/to/llama-server.log \
  --log-prefix \
  --log-timestamps \
  --verbosity 4 \
  -m /path/to/model.gguf
```

Then run:

```bash
python3 llama_runtime_probe.py \
  --base-url http://127.0.0.1:18080 \
  --output-dir runtime-probe-runs \
  --label mixtral-runtime \
  --model dolphin-mixtral \
  --suite-path data/mixtral_probe_prompts.json \
  --max-prompts 4 \
  --repeats 2 \
  --log-file-path /path/to/llama-server.log
```

From the repository root, the guarded live baseline runner checks the server
observability endpoints before launching prompt traffic:

```bash
python3 scripts/run_live_baseline.py --dry-run
python3 scripts/run_live_baseline.py --base-url http://127.0.0.1:18080 --model dolphin-mixtral --preflight-only --preflight-timeout-seconds 2
```

Details live in [live-baseline-runner.md](docs/live-baseline-runner.md).

This path still does not expose semantic expert ids. It can correlate request timing, server metrics, slot state, properties, and log growth.

### Tier 2B: Forward-Hook Semantic Probe

Use this only when the model runtime still exposes Python modules and `register_forward_hook()` works:

```python
from pathlib import Path
import moe_forward_probe

config = moe_forward_probe.ForwardHookProbeConfig(
    output_dir=Path("forward-probe-runs"),
    label="local-hookable-moe",
)
probe = moe_forward_probe.ForwardHookMoEProbe(config=config)
hook_count = probe.attach(model)

with probe.span({"prompt_id": "case-001", "family_id": "code_python"}):
    _ = model(**inputs)

probe.close()
```

The forward-hook path can capture routed expert ids, expert weights or probabilities, routing entropy, per-layer hit counts, and window summaries when the backend exposes those values.

From the repository root, use the hook trace planner before touching a real
local model:

```bash
python3 scripts/plan_hook_trace_capture.py \
  --model-path /path/to/local/transformers-moe \
  --label local-hookable-moe \
  --json
```

The planner emits a synthetic hook smoke command, a guarded
`run_transformers_forward_probe.py --dry-run` command, and a deferred approved
local trace command. The plan itself does not claim semantic expert ids; only a
completed hook trace bundle with nonzero `hook_count`, `router_events.jsonl`,
and `summary.json` router events can support that claim.

## Fixtures And Docs

- [data/mixtral_probe_prompts.json](data/mixtral_probe_prompts.json): prompt corpus for Mixtral-style routing and runtime probes.
- [data/synthetic_controller_trace.json](data/synthetic_controller_trace.json): replay fixture for controller/advisor logic.
- [data/toy_workload.json](data/toy_workload.json): simulator workload.
- [docs/model-target-test-plan.md](docs/model-target-test-plan.md): target classes and live-test command gates.
- [docs/model-plane-manifest-consumption.md](docs/model-plane-manifest-consumption.md): how agents consume Model Plane manifests and select runtime baseline, passive sidecar, or hookable semantic paths.
- [docs/expert-paging-roadmap.md](docs/expert-paging-roadmap.md): phase roadmap from harness/probes to a future expert-paging actuator.
- [docs/expert-paging-step-breakdown.md](docs/expert-paging-step-breakdown.md): small-step execution checklist for roadmap phases, evidence, and blockers.
- [docs/hookable-progression.md](docs/hookable-progression.md): intended progression from passive external observation to internal hookable semantic tracing before fork/controller work.
- [docs/llama-cpp-engine-hook-track.md](docs/llama-cpp-engine-hook-track.md): correction and contract for llama.cpp engine-level MoE router hooks.
- [docs/llama-cpp-engine-hook-traces-2026-06-25.md](docs/llama-cpp-engine-hook-traces-2026-06-25.md): first successful patched llama.cpp Mixtral and Qwen3 router traces.
- [docs/phase-2-repeat-protocol.md](docs/phase-2-repeat-protocol.md): repeat gates for semantic routing traces, including matrix validation, GGUF preflight, and full trace acceptance criteria.
- [docs/flash-moe-inspiration-next-step.md](docs/flash-moe-inspiration-next-step.md): architecture lessons from Flash-MoE for offline expert inventory, dry-run packing, replay metrics, and launch-card gates.
- [docs/pc-llama-debug-mixtral-hook-smoke-2026-06-26.md](docs/pc-llama-debug-mixtral-hook-smoke-2026-06-26.md): PC Ollama Mixtral read-only llama.cpp debug route-tensor smoke.
- [docs/pc-llama-cpp-router-traces-2026-06-26.md](docs/pc-llama-cpp-router-traces-2026-06-26.md): validated PC Ollama Mixtral, Nemotron Cascade, and Qwen3 Coder traces through the patched direct llama.cpp runner.
- [docs/pc-large-gguf-moe-preflights-2026-06-26.md](docs/pc-large-gguf-moe-preflights-2026-06-26.md): metadata-only preflights for oversized PC Llama 4 Scout and DeepSeek V3.1 GGUF blobs.
- [docs/pc-ollama-moe-inventory-refresh-2026-06-26.md](docs/pc-ollama-moe-inventory-refresh-2026-06-26.md): current PC Ollama MoE inventory and per-model hook attempt outcomes.
- [docs/hookable-moe-attempts-pc-gx10-2026-06-25.md](docs/hookable-moe-attempts-pc-gx10-2026-06-25.md): current PC/GX10 hookability attempt matrix summary.
- `scripts/validate_llama_cpp_router_trace.py`: dependency-free validator for patched llama.cpp router trace JSONL.
- `scripts/gguf_moe_preflight.py`: dependency-free GGUF metadata/tensor-table preflight for large MoE blobs without loading tensor data.
- `docker/llama-moe-router-trace.Dockerfile`: builds the patched direct llama.cpp JSONL trace binary for local GGUF/Ollama blobs.
- [data/expert_paging_roadmap.json](data/expert_paging_roadmap.json): machine-readable roadmap and actuator-spike checklist. Validate it from the repository root with `python3 scripts/plan_expert_paging.py`.
- [data/phase3_real_evidence_bundle.fixture.json](data/phase3_real_evidence_bundle.fixture.json): valid-but-not-ready Phase 3 bundle manifest fixture for trace, inventory, optional policy-candidate trace receipt, optional fallback, optional live proof, and approval metadata.
- `scripts/plan_baseline_policy_replay.py`: offline Phase 3 replay planner for observe-only, keep-hot, preload-shortlist, evict-cold, fallback-dense, and no-live-actuator policies over joined trace+inventory artifacts.
- `scripts/plan_expert_store_layout.py`: dry-run expert-store layout planner that estimates packed target files, source byte ranges, required disk, and missing components without writing tensor data.
- `scripts/plan_real_model_trace_inventory_pairing.py`: Phase 3 gate that distinguishes fixture-only trace/inventory joins from scanner-derived real-model evidence.
- `scripts/plan_dense_fallback_comparison.py`: validates an optional dense/full-runtime fallback comparison artifact and only marks it ready when builder provenance shows receipt-ready, pair-consistent managed/dense inputs, or records the current missing-output blocker without launching a runtime.
- `scripts/build_dense_fallback_comparison.py`: builds the dense/full-runtime fallback comparison artifact from already-saved managed and dense output summaries only when both inputs have ready capture receipts from the same request, prompt set, model, and prompt family, or emits a valid not-ready template.
- `scripts/build_phase3_output_summary.py`: builds and validates managed/dense saved-output templates with a capture receipt; filled rows only become ready after the receipt records the approved request, host/backend, timestamp, and approval flags.
- `scripts/phase3_trace_receipts.py`: validates candidate router-trace capture receipts that bind the trace to the approved request, prompt set, model/backend, host, timestamp, and approval flags.
- `scripts/build_phase3_trace_receipt.py`: builds and validates fillable not-ready candidate trace receipt templates beside each planned candidate router trace.
- `scripts/plan_phase3_policy_candidate_trace.py`: turns one real-evidence bundle into repeated prompt-set, candidate-router-trace, trace receipt, replay-validation, and updated-bundle requests, while keeping explicit planned trace paths valid before capture for the policy-candidate blocker without running prompts; `--output-md path/to/handoff.md` writes a Markdown handoff with artifact blockers, trace-receipt contract, and validation/build commands.
- `scripts/plan_phase3_dense_fallback_capture.py`: turns one real-evidence bundle into prompt-set, managed-output, dense-output, comparison-artifact, and updated-bundle requests for the dense fallback blocker without running prompts, while binding planned output paths before capture; `--output-md path/to/report.md` writes a Markdown handoff with artifact blockers, runtime-capture contract, and build/validation commands.
- `scripts/plan_phase3_go_no_go.py`: aggregates Phase 3 gates, including repo-local matrix as the real-model pairing basis, handoff scaffold coverage, runtime-capture request audit, policy-candidate trace receipt readiness, and capture-result intake receipt gate and receipt-fill manifests, into an explicit no-go/go decision packet for live expert-paging work with policy-candidate blocked-bundle counts, no-reuse-distance and prompt-identity blocker counts, launch-card template/binding/runtime-command counts, runtime-actuator spike proof counts, and the next candidate-trace operator step; `--output-md path/to/decision.md` writes a Markdown report with gate status, no-go reasons, input evidence, next actions, and safety contract.
- `scripts/plan_phase3_evidence_packet.py`: one-command Phase 3 audit packet that indexes offline evidence, the repo-local real-evidence matrix, matrix-aware pairing status, scaffold coverage, saved-request drift, runtime-capture queue selection, approval metadata rebuild command manifest, parity between pre-capture and capture-result approval manifests, receipt-requirement parity between runtime requests and intake gates, path-aware operator-queue parity, recommended and all-request post-approval capture-fill parity between request audit artifacts and intake next steps, all-request capture queue manifest, all-request approval command manifest, all-request validator command manifest, all-request downstream handoff manifest, all-request receipt-fill manifest, all-request receipt-fill command manifest, all-request receipt/validator parity, receipt-command validator coverage, validator-manifest runtime/future coverage, runtime-actuator design readiness, runtime-actuator spike proof handoff, Phase 3 blocker closure manifest with mapping-ready/evidence-complete status, Phase 3 blocker evidence ledger with row-level missing-evidence counts, Phase 3 blocker resolution queue with ordered work packages, dependency order, completion gates, runtime-actuator proof handoff fields, capture-result intake drift, receipt-gate counts, receipt-fill manifest counts, and approval transition preview counts, post-approval capture-fill plan counts, policy-candidate blocker evidence, an ordered promotion checklist, the recommended runtime-capture sequence, recommended post-approval preview, recommended runtime-capture preflight manifest, recommended runtime-capture command contract, recommended runtime-capture execution coverage, all-request runtime-capture execution coverage, repo-level launch-card library and binding handoff status, recommended filled launch-card binding intake, recommended and all-request runtime-capture runtime-command/manual execution coverage, all-request manual capture runbook with source-request/prompt-set/approval-contract counts, all-request post-capture intake runbook, recommended post-approval capture-fill plan, recommended policy-candidate trace handoff, recommended dense fallback capture handoff, recommended live capability proof handoff, recommended and all-request validator command manifests, no-go reason closure mapping with mapping-ready/evidence-complete ready-scope status, blocker evidence ledger row counts, remaining gaps, optional live proof, and the policy-candidate trace receipt required before Phase 3 completion; add `--policy-candidate-trace-receipt-path path/to/receipt.json` when validating a filled candidate trace, `--output-md path/to/report.md` to write a Markdown report, `--output-runtime-capture-launch-card path/to/card.json` to export the recommended planned-only launch-card template, `--runtime-capture-launch-card path/to/filled-card.json` to fold one filled Model Plane command binding into recommended execution coverage, `--runtime-capture-launch-card-dir path/to/filled-cards` to fold a repo-wide filled-card batch into all-request execution coverage, `--output-operator-handoff-dir path/to/dir` writes a metadata-only handoff package with the evidence packet, README, binding worksheet, Model Plane artifact-writer contract request, recommended planned launch-card template, recommended runtime-capture preflight, recommended runtime-capture command contract, recommended runtime-capture execution coverage, all-request runtime-capture execution coverage, runtime-capture launch-card directory manifest, reuse-evidence capture plan, manual capture runbook, post-capture intake runbook, capture queue, approval command manifest, validator manifest, downstream handoff manifest, receipt-fill manifests, blocker-closure manifest, blocker-evidence ledger, and blocker-resolution queue; and --live-proof-artifact-path path/to/live-proof.json when a residency/control proof artifact exists.
- `scripts/plan_phase3_operator_handoff.py`: validates a metadata-only operator handoff package, or generates a temporary current package when no directory is provided, checking required files, safety flags, worksheet/card schemas, recommended and all-request approval command readiness, approval-command-manifest/capture-queue parity, recommended capture-queue rank/path/next-artifact agreement, recommended launch-card worksheet/template/capture-queue task parity, Model Plane artifact-writer contract request/schema/task parity, selected preflight packet/file parity, selected command-contract packet/file parity, selected execution-coverage packet/file parity, all-request execution-coverage packet/file parity, runtime-capture launch-card directory packet/execution/handoff parity with external staging directories allowed but repo-relative request and source-card paths checked, reuse-evidence capture-plan packet/gap/handoff/recommended-capture parity, manual-capture runbook packet/file parity, post-capture intake runbook packet/file parity, post-capture artifact-gate/manual-task/receipt-command parity, manual-task/receipt-command parity, manual-task source-request/prompt-set/approval-contract counter parity, all-request binding-worksheet/capture-queue task parity, repo-relative artifact/request/receipt path safety, capture queue counts, validator command counts, validator-manifest runtime/future artifact coverage, receipt-fill counts, receipt-command/capture-queue artifact-key parity, receipt-command validator coverage against capture-queue validator counts plus source-request/prompt-set/approval-contract readiness, downstream handoff request/section/path coverage, blocker-closure reason/action/path coverage, blocker closure counts, blocker-evidence ledger row/count/packet parity, blocker-resolution queue package/row/dependency/completion-gate/validator parity plus runtime-actuator proof-handoff counts, requirement coverage, and control-dependency checks, and README markers without running runtimes, Docker, endpoints, secrets, or prompt traffic.

- `scripts/plan_phase3_real_evidence_capture.py`: turns the evidence packet gaps into real trace, trace receipt, scanner inventory, dense fallback, replay, and live-capability artifact requests without running models or prompts; `--output-md path/to/handoff.md` writes a Markdown handoff with request statuses, planned outputs, validator commands, blocking requests, and safety contract.
- `scripts/plan_phase3_runtime_actuator_design.py`: validates the selected backend actuator design boundary from the managed-loading plan, keeps residency observation/control and cleanup gaps explicit, and refuses live-actuator readiness without touching a runtime.
- `scripts/plan_phase3_runtime_actuator_spike.py`: turns the selected backend actuator gaps into a concrete proof handoff with patch/probe boundaries, proof artifacts, dependency order, and completion gates before any residency-control spike is allowed.
- `scripts/plan_phase3_live_capability_proof.py`: validates the future live residency observation/control, cleanup/restore, artifact-export, existing source-bundle binding, expected bundle/model/backend context, and existing exported artifact paths once export status is available, without touching a runtime; `--output-md path/to/handoff.md` writes a Markdown handoff with required proof sections, blockers, context binding, and safety contract.
- `scripts/build_phase3_live_capability_proof_template.py`: builds a fillable valid-but-not-ready proof template for residency observation/control, cleanup/restore, artifact export, source bundle, and prompt context without touching a runtime. The generated runtime-capture requests reference these templates as future adapter-bound fills.
- `scripts/plan_phase3_artifact_intake.py`: scans local/generated router trace artifacts, matches optional GGUF inventory summaries as hints, emits optional Docker/Ollama-volume scanner commands for traced blobs, and reports the exact scanner-manifest and fallback artifacts still needed for Phase 3 handoff.
- [phase3-real-evidence/README.md](phase3-real-evidence/README.md): repo-local Phase 3 real-evidence matrix with six validated trace/inventory/replay bundles, policy-candidate blocker counts, trace-receipt required/ready/blocked counts, no-reuse-distance and prompt-identity diagnostics, and no remaining scanner coverage-gap artifacts.
- `scripts/plan_phase3_real_evidence_bundle.py`: validates a Phase 3 handoff bundle that binds trace, optional policy-candidate trace receipt, scanner inventory, policies, managed-loading plan, approvals metadata, optional fallback comparison evidence, and optional live capability proof; attached live proof must match the bundle model, backend, prompt family, and source bundle path, and ready export paths must exist.
- `scripts/plan_phase3_real_evidence_matrix.py`: summarizes repo-local Phase 3 real-evidence bundles, real pair readiness, replay validity, policy-candidate trace-receipt readiness counts, remaining blockers, and scanner coverage/profile status.
- `scripts/plan_phase3_runtime_capture_request.py`: audits saved runtime-capture request artifacts, rebuilds expected request state from local metadata, reports path/status/approval/embedded-summary/source/validator-command drift, summarizes required validator-command coverage across candidate trace, managed output, dense output, and live-proof artifacts, recommends the first approval-bound capture request with queue rank and selection rationale, emits an approval metadata rebuild command plus a compact approval command manifest for recording approved booleans, emits a metadata-only post-approval preview proving the recommended request becomes ready-for-operator without capture completion or mutation, and emits an ordered approval/fill sequence with missing approval flags, trace-receipt targets, per-artifact capture receipt requirements, and a capture-queue selection summary with approval-required, ready-for-operator-capture, and approved-but-incomplete counts without running prompts; `--output-md path/to/handoff.md` writes a pre-capture operator handoff report.
- `scripts/plan_phase3_runtime_capture_commands.py`: turns the selected saved request into a planned-only Model Plane or launch-card command contract for candidate-router traces and managed/dense output fills; it records prompt-set, artifact, receipt, request, approval, validator, and missing command-binding requirements without launching a runtime, running Docker, reading secrets, or sending prompt traffic; `--output-launch-card path/to/card.json` writes a planned-only launch-card template for Model Plane binding, and `--launch-card path/to/filled-card.json` validates a filled card offline, promoting runtime-command readiness only when every task has exact output and receipt paths, the exact prompt-set binding, approval gates, prompt-traffic acknowledgement, the exact runtime-capture request path, and real callable ids or launch commands.
- `scripts/plan_phase3_launch_card_library.py`: indexes all repo-local Phase 3 runtime-capture launch cards, reports template/model-plane/binding/runtime-command readiness, exposes a repo-wide 18-task binding handoff worksheet with exact prompt-set/artifact/receipt/request paths, prompt-traffic acknowledgement, and validation commands, writes a metadata-only Model Plane artifact-writer contract request with `--output-model-plane-contract-request`, counts the 18 current unbound runtime command slots, writes a fillable worksheet with `--output-binding-worksheet`, validates a filled worksheet with `--binding-worksheet` or a returned Model Plane fulfillment with `--model-plane-contract-fulfillment`, can emit filled launch cards with `--output-filled-card-dir`, and can write a Markdown operator report without executing runtimes or prompt traffic.
- `scripts/plan_phase3_capture_result_intake.py`: audits saved runtime-capture requests after local capture artifacts are filled, requires the saved request to be drift-free, reports aggregate drift-free/drifted saved-request counts, ready-for-operator and approved-but-incomplete request counts, runtime-approval-missing versus approved-runtime-capture-pending counts, aggregate capture-receipt gate coverage, flat receipt-fill manifest readiness, output receipt binding readiness, and metadata-only approval rebuild command availability plus a compact approval command manifest for missing-approval runtime steps, emits a metadata-only approval transition preview showing which request becomes ready-for-operator while receipt fills remain missing, emits a post-approval capture-fill plan with artifact/receipt paths, after-approval runtime statuses, missing-fill counts, and validator-command coverage, verifies candidate trace receipts and managed/dense output receipts are bound to the same saved request and prompt set with recorded approvals before promotion, reuses the policy-candidate, dense fallback, live-proof, and bundle validators, and reports whether a bundle update, Phase 4 candidate, or live-spike candidate is actually supported, plus the next operator step for each request; `--show-queue` prints a compact per-request queue and `--output-md path/to/report.md` writes a durable Markdown handoff report with detailed per-request queues plus validator/build command blocks without dumping JSON.
- `scripts/plan_phase3_handoff_coverage.py`: validates that each real Phase 3 bundle has local prompt-set, candidate-trace-receipt, output-summary, runtime-capture-request, planned-only runtime-capture launch-card template, and live-proof-template scaffolding before approved runtime capture; `--output-md path/to/report.md` writes a Markdown coverage and repair-command report.
- `scripts/build_phase3_real_evidence_bundle.py`: builds the Phase 3 handoff bundle manifest from selected artifact paths, including optional policy-candidate trace receipt, fallback, and live-proof paths, plus explicit approval metadata without running models or prompts.
- `memory-moe-mvp/tests/test_phase3_positive_path.py`: proves generated non-fixture trace, trace receipt, inventory, fallback, bundle, packet, and capture-plan artifacts can promote Phase 3 to Phase 4-ready without claiming live paging, and proves that promotion stays blocked when the trace receipt is missing.
- [docs/managed-expert-loading.md](docs/managed-expert-loading.md): first managed expert loading contract pass, including observe-only, replay/simulate, and future live-actuator boundaries.
- [data/managed_expert_loading_plan.json](data/managed_expert_loading_plan.json): machine-readable managed-loading contract. Validate it from the repository root with `python3 scripts/plan_managed_expert_loading.py`.
- [docs/edge-hardware-quickstart.md](docs/edge-hardware-quickstart.md): feasibility tiers for CPU, GPU, Apple Silicon, Jetson/ARM, and Android phone/emulator targets, with small dense models treated as near-term routable experts.
- [data/edge_hardware_quickstart.json](data/edge_hardware_quickstart.json): machine-readable edge hardware and small-model routing matrix. Validate it from the repository root with `python3 scripts/plan_edge_hardware_quickstart.py`.
- [../MODEL_CARD.md](../MODEL_CARD.md): model-card style scope, target-class, safety, and limitation summary for the harness.
- [docs/portability-and-gpu-hosts.md](docs/portability-and-gpu-hosts.md): portable Tier 0 path, host preflight, GPU expectations, and live-machine dependencies.
- [docs/probe-observability-notes.md](docs/probe-observability-notes.md): probe surfaces and `llama.cpp` observability knobs.
- [docs/controller-architecture.md](docs/controller-architecture.md): controller pattern and failure-mode guardrails.

Historical probe result fixtures remain under `probe-results/`.

## Local Checks

From the repository root, run the upload-readiness check:

```bash
python3 scripts/check_project.py
```

To inspect a machine before live GPU tests:

```bash
python3 scripts/check_host.py
```

Use `python3 scripts/check_host.py --require-gpu` only when the next step truly requires visible NVIDIA/CUDA or AMD/ROCm tooling. Tier 0 does not require a GPU.

From this package directory, the direct component checks are:

```bash
python3 model_target_registry.py
python3 -m unittest discover -s tests
python3 -m py_compile \
  memory_moe.py \
  moe_shared_contract.py \
  moe_forward_probe.py \
  run_forward_probe_demo.py \
  moe_controller_demo.py \
  llama_runtime_probe.py \
  llama_sidecar.py \
  model_target_registry.py
```

## Docker Materials

Docker files are present but not part of the local readiness command:

- [docker/llama-sidecar.Dockerfile](docker/llama-sidecar.Dockerfile)
- [docker/llama-with-sidecar-entrypoint.sh](docker/llama-with-sidecar-entrypoint.sh)
- [docker/forward-hook-probe.Dockerfile](docker/forward-hook-probe.Dockerfile)
- [docker/forward-hook-probe.requirements.txt](docker/forward-hook-probe.requirements.txt)
- [docker/forward-hook-probe.compose.yaml](docker/forward-hook-probe.compose.yaml)

The forward-hook container stack is for PyTorch/Transformers-style runtimes, not stock `llama.cpp`.

## Windows Helper

[run_mixtral_probe.ps1](run_mixtral_probe.ps1) is a Windows PowerShell helper for the older direct Mixtral prompt probe. Its defaults are relative to this package directory:

```powershell
py -3 .\run_mixtral_probe.ps1
```

The maintained cross-platform live command shapes are in [docs/model-target-test-plan.md](docs/model-target-test-plan.md).

## Next Work

The grounded next step is the hookable semantic trace layer: keep the passive
sidecar as external runtime evidence, run the synthetic hook smoke after
hookable-probe changes, select one small local or cloud hookable MoE target, and
run `scripts/plan_hook_trace_capture.py` before approved local hook execution.

For edge devices, the lower-risk path is still to compare small dense models as
routable experts first, then revisit true expert paging only after hookable
traces and runtime actuator evidence exist.
