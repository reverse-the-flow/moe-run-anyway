# Expert Paging Roadmap

MoE Run Anyway is the harness and contract layer for expert-paging work. It can
plan probes, collect comparable artifacts, replay routing traces, and audit
controller policies. It is not yet a runtime actuator.

The current objective is to move from harness/probes to real expert paging
without pretending the missing runtime control exists. The practical next
abstraction is managed expert loading: explicit expert inventory discovery,
semantic routing visibility requirements, runtime capability detection,
residency state, policy decisions, backend adapter boundaries, dense fallback,
cleanup, and `memory-moe-bridge-v1` artifact compatibility.

For execution tracking, use
[expert-paging-step-breakdown.md](expert-paging-step-breakdown.md). It splits
each phase into small steps with current status, evidence, and blockers.

The intended progression is passive external observation, then internal
hookable semantic tracing, then fork/runtime-actuator or controller work only
after hookable evidence. See
[hookable-progression.md](hookable-progression.md) for that boundary.

The verified build-up is deliberately slower than "trace, then fork." The next
stair is semantic trace, offline expert inventory, dry-run storage/layout
planning, replay with byte and churn accounting, backend-adapter feasibility,
and only then a guarded live actuator. The Flash-MoE inspiration note captures
why this matters: clever cache and prefetch ideas should be rejected or accepted
by replay evidence before runtime work.

The managed loading contract lives in
[managed-expert-loading.md](managed-expert-loading.md) and
[managed_expert_loading_plan.json](../data/managed_expert_loading_plan.json).
Validate it with `python3 scripts/plan_managed_expert_loading.py`. The likely
future path may require a small `llama.cpp` controller patch or fork if evidence
shows `llama.cpp` is the right substrate.

The saved Model Plane handoff fixture is
[model_plane_moe_probe_manifest.runtime_baseline.fixture.json](../data/model_plane_moe_probe_manifest.runtime_baseline.fixture.json).
Validate the Phase 0 handoff gate with:

```bash
python3 scripts/plan_moe_probe_manifest.py memory-moe-mvp/data/model_plane_moe_probe_manifest.runtime_baseline.fixture.json --json
```

The Phase 1 runtime baseline artifact contract is
[runtime_baseline_artifact_contract.json](../data/runtime_baseline_artifact_contract.json).
Validate it with `python3 scripts/plan_runtime_baseline_artifacts.py`.
Plan saved-manifest baseline capture packets with
`python3 scripts/plan_runtime_baseline_capture.py --json`; the default
llama.cpp fixture and
[model_plane_moe_probe_manifest.vllm_runtime_baseline.fixture.json](../data/model_plane_moe_probe_manifest.vllm_runtime_baseline.fixture.json)
exercise the llama.cpp and OpenAI-compatible planning paths without endpoint
checks or prompt traffic.

## Current Truth Table

| Surface | Exists now | What it can prove | What it cannot prove |
| --- | --- | --- | --- |
| Model Plane MoE manifest handoff | Yes | Run-scoped model, backend, endpoint, log, observability, and safety hints can be exported for MoE planning. | It does not expose semantic expert ids unless the runtime does. |
| Managed expert loading contract | Yes | Expert inventory, routing visibility, residency state, policy decision, backend adapter, fallback, cleanup, and artifact requirements are machine-checkable. | It cannot read or mutate live expert residency. |
| Runtime baseline planner | Yes | Safe dry-run and preflight-only commands for `llama.cpp`, vLLM, Ollama, or OpenAI-compatible endpoints. | It does not launch servers, send prompt traffic, or page experts. |
| Runtime baseline artifact contract | Yes | Phase 1 baseline artifact classes and evidence labels can be validated offline. | It does not collect live artifacts or prove semantic expert ids from stock endpoint telemetry. |
| Passive sidecar | Yes | Request-boundary telemetry and upstream observability can be captured non-invasively. | It cannot alter expert residency or infer semantic routing from endpoint timing alone. |
| Stock `llama.cpp` runtime probe | Yes | Metrics, slots, props, timings, and optional log growth can become runtime evidence. | Stock endpoint telemetry is not semantic expert routing. |
| Forward-hook PyTorch probe | Yes for hookable runtimes | Router outputs, routed expert ids, weights, entropy, and per-layer hit counts can be captured when modules expose them. | It does not apply to stock `llama.cpp` GGUF execution. |
| Patched direct `llama.cpp` router trace | Yes for direct GGUF runs | Selected expert ids and selected weights can be emitted as `memory-moe-bridge-v1` JSONL, including read-only Ollama blobs. | It does not make stock Ollama API traffic emit traces, control residency, or prove paging. |
| Offline expert inventory | Yes for schema fixtures, safetensors headers, and GGUF tensor tables | Repo-local inventory manifests validate layer/expert/component rows, `scripts/scan_safetensors_expert_inventory.py` can dry-run safetensors headers or shard indexes, and `scripts/scan_gguf_expert_inventory.py` can dry-run GGUF metadata/tensor tables into that schema without loading tensor values. | It does not load/page/mutate expert residency, write packed stores, or prove live paging. |
| Controller replay/advisor | Yes | Offline residency-policy scoring and dense-fallback comparison over traces or summaries. | It cannot pin, evict, preload, or load only routed experts in a live backend. |
| Expert paging simulator | Yes | Logical policy behavior, churn, warm-hit rate, and replay shape. | It is not live tensor residency control. |
| Runtime actuator | No | Not applicable. | No live expert loading, eviction, pinning, preloading, or offload-behavior override exists yet. |

## Roadmap Phases

### Phase 0: Harness Contract And Model Plane Handoff

Status: complete. The roadmap, managed-loading contract, saved Model Plane
manifest fixture, and manifest planner handoff gate validate locally without
runtime side effects.

Goal: make the current boundary explicit and machine-checkable.

Deliverables:

- Model Plane run-scoped MoE manifests remain the durable handoff contract.
- MoE planners consume manifests without launching models, downloading weights,
  authenticating, running Docker, or sending prompt traffic.
- Managed expert loading requirements are explicit and machine-checkable:
  inventory discovery, semantic routing visibility, runtime capability
  detection, residency states, policy decisions, backend adapter boundaries,
  fallback, cleanup, and artifact compatibility.
- A saved Model Plane manifest fixture exercises the runtime-baseline handoff
  path with `--json`.
- A planned `harness_run_request` stage describes what the harness would run
  after user approval, but remains planning-only.
- Roadmap and actuator-spike requirements are tracked as artifacts.

Evidence gate:

- `scripts/plan_moe_probe_manifest.py` and `scripts/plan_expert_paging.py`
  validate local artifacts and print next actions without side effects.
- `scripts/plan_managed_expert_loading.py` validates the managed loading
  contract and reports missing capabilities before any runtime work.
- Dependency-free unit tests cover the roadmap and planner outputs.

### Phase 1: Artifact Evidence From Runtime Baselines

Status: blocked on approved live upstream artifacts. The contract/planning
layer exists, and passive sidecar implementation exists, but no live runtime has
been launched and no prompt traffic is part of this phase start.

Goal: collect comparable evidence from existing runtimes before designing an
actuator.

Deliverables:

- Passive sidecar artifacts from an approved running upstream.
- llama.cpp baseline artifacts from `/metrics`, `/slots`, `/props`, response
  timings, optional log growth, and Model Plane metadata.
- vLLM and OpenAI-compatible baseline artifacts where readiness endpoints exist.
- A runtime baseline artifact contract validated by
  `scripts/plan_runtime_baseline_artifacts.py`.
- Artifact shape mapped to `memory-moe-bridge-v1`.
- Capability labels for managed loading: inventory, routing visibility,
  residency read/write, fallback, artifact export, and cleanup.
- Clear labels for runtime evidence versus semantic routing evidence.

Evidence gate:

- At least one sidecar artifact bundle or runtime baseline artifact bundle from
  an approved running backend.
- At least one run-scoped artifact bundle per backend class.
- Runtime baseline artifact classes and evidence labels validate offline.
- Baseline artifacts show reproducible timing and observability fields.
- No artifact claims semantic expert ids unless the runtime explicitly exposes
  router outputs.

### Phase 2: Semantic Routing Trace Capture

Status: blocked for the remaining Python hookable-runtime lane, with the direct
GGUF trace path validated. The synthetic hook smoke path exists, the patched
direct `llama.cpp` trace path has produced validated GX10 and PC traces, and
the Python hookable-runtime lane remains open for a suitable Transformers target.

Goal: capture semantic routing traces from hookable internals before any
residency actuator or fork/controller work.

Deliverables:

- The synthetic no-model hook smoke path remains green.
- A real local hookable trace runner for a PyTorch/Transformers-style MoE.
- A patched direct `llama.cpp` router trace path for GGUF and read-only Ollama
  blobs.
- Router output capture for layer id, expert id, score or probability, entropy,
  prompt family, and token/window metadata.
- A repeat protocol that separates matrix validation, metadata preflight, and
  full semantic trace acceptance.
- Compatibility with replay/controller artifacts.

Evidence gate:

- Dense fallback output can be recorded or compared for the same prompt set.
- Trace artifacts validate under `memory-moe-bridge-v1`.
- Hooking failure modes are explicit and do not silently downgrade to timing
  inference.
- Empty-JSONL timeouts are recorded as runtime-budget blockers, not as semantic
  trace successes or model incompatibility.

### Phase 3: Inventory-Grounded Replay Audit

Status: in progress. Six repo-local trace/inventory/replay bundles now validate as real-model evidence; no bundle is policy-candidate-ready yet, and fallback comparison, live residency observation/control, and cleanup/restore proof remain missing.

Goal: prove or reject managed-loading policies offline before touching live
runtime residency. Replay must combine semantic traces with expert inventory,
estimated bytes, reuse distance, miss rate, churn, fallback behavior, and
explicit rejection reasons.

Deliverables:

- Expert inventory manifest fixture, validator, safetensors header/index scanner dry run, GGUF tensor-table scanner dry run, trace-to-inventory replay join planner, real-model trace/inventory pairing gate, baseline policy definitions, and baseline policy replay planner.
- Dry-run expert-store layout plan that estimates disk and missing components
  without writing tensor data.
- Trace-to-inventory planner joins semantic traces to inventory-derived byte
  costs; replay controller evaluates candidate residency policies over that
  joined artifact.
- Dense fallback comparison, a local comparison artifact builder, receipt-ready pair-consistent comparison provenance, or an explicit fallback-output blocker report is part of the audit contract.
- Baseline replay policy definitions use the managed-loading vocabulary:
  `observe_only`, `preload`, `pin`, `keep`, `evict`, `demote`,
  `fallback_dense`, `abort_run`, required metrics, and rejection reasons.
- Failure-mode reports cover churn, missed experts, byte misses, fallback
  frequency, quality deltas, rejected policies, the explicit Phase 3 go/no-go decision with optional Markdown export, a one-command evidence packet with optional Markdown export and metadata-only operator handoff package export and validation, ordered promotion checklist, recommended runtime-capture sequence, recommended policy-candidate trace handoff, recommended dense fallback capture handoff, recommended live capability proof handoff, runtime-actuator design readiness, runtime-actuator spike proof handoff, recommended and all-request validator command manifests, optional live-proof attachment, repo-local matrix, matrix-aware pairing status, scaffold, request-audit, approval-manifest parity, receipt-requirement parity, path-aware operator-queue parity, recommended and all-request post-approval capture-fill parity, recommended runtime-capture preflight manifest, recommended runtime-capture command contract, recommended and all-request runtime-capture runtime-command/manual execution coverage, all-request manual capture runbook with source-request/prompt-set/approval-contract counts, all-request post-capture intake runbook, all-request capture queue manifest, all-request approval command manifest, all-request validator command manifest, all-request downstream handoff manifest, all-request receipt-fill manifest with candidate-router trace, managed-output, and dense-output class counts, all-request receipt-fill command manifest, all-request receipt/validator parity, receipt-command validator coverage, validator-manifest runtime/future coverage, Phase 3 blocker closure manifest with mapping-ready/evidence-complete status, Phase 3 blocker evidence ledger with row-level missing-evidence counts, Phase 3 blocker resolution queue with ordered work packages, next-unblocked work-package pointer, dependency order, completion gates, embedded runtime-actuator proof handoff, and closure-level validator counts, and intake receipt gates, a real-evidence capture plan with optional Markdown handoff, a prompt-set builder, output-summary templates with capture receipts, trace receipt validation/templates, runtime-capture requests, planned-only runtime-capture launch-card templates, a launch-card library planner with a binding handoff worksheet and Model Plane artifact-writer contract request, a saved-request audit with optional Markdown operator handoff export, a capture-result intake audit with a flat receipt-fill manifest split by candidate-router trace, managed-output, and dense-output artifact classes, a post-approval capture-fill plan, and per-request next-step queueing, live-capability proof templates, a handoff coverage planner with optional Markdown repair report, a policy-candidate trace planner, a matrix-aware top-level decision gate, plus a bundle builder/validator pair for Phase 4 handoff.

Evidence gate:

- Expert inventory validates for at least one fixture, safetensors scanner output, or GGUF scanner output.
- Dry-run packer/layout plan validates without writing tensors.
- Trace/inventory join reports unique experts, estimated bytes, reuse
  distance, and whether the pair is fixture-only or scanner-derived real-model evidence; policy replay reports miss rate, churn, fallback frequency, and
  rejected-policy reasons.
- Controller improves or explains warm-hit rate without unacceptable churn, or
  the report recommends no live actuator yet.
- Dense fallback comparison bounds quality or behavior drift; the builder now requires ready capture receipts with existing source request and prompt-set bindings before writing paired comparison artifacts, the validator only marks artifacts ready when pair-consistent builder provenance is present, non-exact quality labels still need review, and `scripts/build_phase3_output_summary.py` plus `scripts/plan_phase3_dense_fallback_capture.py` expose saved-output artifacts, capture receipts, Markdown handoff, and commands needed per real bundle after the shared prompt set exists.
- Real-evidence capture planning names the required trace, inventory, pairing, fallback, replay, and live-capability artifacts before Phase 4 work starts and can write a Markdown handoff with request statuses, planned outputs, validator commands, blocking requests, and safety contract; `scripts/build_phase3_prompt_set.py` builds the shared repeated prompt set; `scripts/build_phase3_output_summary.py` builds valid not-ready managed/dense output templates with capture receipts; `scripts/phase3_trace_receipts.py` validates candidate router-trace receipts; `scripts/build_phase3_trace_receipt.py` creates fillable not-ready receipt templates; `scripts/build_phase3_runtime_capture_request.py` binds approval-bound runtime fills, the future live-proof fill, validator commands, and trace-receipt templates; `scripts/plan_phase3_runtime_capture_request.py` audits saved request path/status/approval/embedded-summary/source/validator-command drift before operator use, summarizes required validator-command coverage, recommends the first approval-bound capture request with missing approval flags, queue rank, and selection rationale, emits an approval metadata rebuild command plus a compact approval command manifest for recording approved booleans, emits a metadata-only post-approval preview proving the recommended request becomes ready-for-operator without capture completion or mutation, emits an ordered approve/capture/fill/intake sequence, renders capture receipt requirements, emits a capture-queue selection summary with approval-required, ready-for-operator-capture, and approved-but-incomplete counts, and can emit a Markdown operator handoff with `--output-md`; `scripts/plan_phase3_runtime_capture_commands.py` builds a planned-only Model Plane or launch-card command contract for candidate-router trace and managed/dense output fills, names prompt-set, artifact, receipt, and request paths plus missing command bindings, can export a planned-only launch-card template, the evidence packet can export the recommended template with `--output-runtime-capture-launch-card`, can validate a filled launch card with `--launch-card` as binding-ready only when every task path, prompt-set binding, approval gate, prompt-traffic acknowledgement, runtime-capture request path, and callable id or launch command matches, and keeps runtime execution blocked until real commands exist; `scripts/plan_phase3_launch_card_library.py` indexes all six planned-only launch cards, exposes the 18-task binding handoff worksheet and Model Plane artifact-writer contract request with exact prompt-set/artifact/receipt paths and validation commands, writes a fillable worksheet with `--output-binding-worksheet`, writes the metadata-only contract request with `--output-model-plane-contract-request`, validates filled worksheets with `--binding-worksheet`, validates returned Model Plane fulfillments with `--model-plane-contract-fulfillment`, requires prompt-traffic acknowledgement plus exact request/prompt-set/artifact/receipt paths, can emit offline-validated filled cards with `--output-filled-card-dir`; the evidence packet can consume that directory with `--runtime-capture-launch-card-dir` to promote all-request execution coverage offline, and keeps template readiness separate from Model Plane binding and runtime-command readiness; the evidence packet consumes that repo-wide library and binding handoff summary before approved capture and can write a metadata-only operator handoff package with the packet, README, binding worksheet, Model Plane artifact-writer contract request, recommended launch-card template, recommended runtime-capture preflight, recommended runtime-capture command contract, recommended runtime-capture execution coverage, all-request runtime-capture execution coverage, runtime-capture launch-card directory manifest, reuse-evidence capture plan, manual capture runbook, post-capture intake runbook, capture queue, approval command manifest, validator manifests, downstream handoff manifest, receipt-fill manifests with artifact-class counts, blocker-closure manifest, blocker-evidence ledger, and blocker-resolution queue, then validates that package with `scripts/plan_phase3_operator_handoff.py`, including recommended and all-request approval-command readiness, approval-command-manifest/capture-queue parity, recommended capture-queue agreement, recommended launch-card worksheet/template/capture-queue task parity, Model Plane artifact-writer contract request/schema/task parity, selected preflight packet/file parity, selected command-contract packet/file parity, selected execution-coverage packet/file parity, all-request execution-coverage packet/file parity, runtime-capture launch-card directory packet/execution/handoff parity with external staging directories allowed but repo-relative request and source-card paths checked, reuse-evidence capture-plan packet/gap/handoff/recommended-capture parity, manual-capture runbook packet/file parity, post-capture intake runbook packet/file parity, post-capture artifact-gate/manual-task/receipt-command parity, manual-task/receipt-command parity, manual-task source-request/prompt-set/approval-contract counter parity, all-request binding-worksheet/capture-queue task parity, repo-relative artifact/request/receipt path safety, receipt-command/capture-queue artifact-key parity, receipt-fill artifact-class count parity, receipt-command validator coverage, validator-manifest runtime/future coverage, downstream handoff request/section/path coverage, blocker-closure reason/action/path coverage, blocker-evidence ledger row/count/packet parity, and blocker-resolution queue package/row/dependency/completion-gate/validator parity, next-unblocked work-package pointer, plus runtime-actuator proof-handoff counts, requirement coverage, and control-dependency checks; the evidence packet separates blocker-closure mapping readiness from evidence completeness so manifest-ready does not imply promotion-ready; `scripts/plan_phase3_capture_result_intake.py` audits filled request artifacts before bundle promotion, requires the saved request to be drift-free, reports aggregate drift-free/drifted saved-request counts, ready-for-operator and approved-but-incomplete request counts, runtime-approval-missing versus approved-runtime-capture-pending counts, aggregate capture-receipt gate coverage, flat receipt-fill manifest readiness split by candidate-router trace, managed-output, and dense-output artifact classes, output receipt binding readiness, and metadata-only approval rebuild command availability plus a compact approval command manifest on missing-approval runtime steps, emits a metadata-only approval transition preview showing which request becomes ready-for-operator while receipt fills remain missing, emits a post-approval capture-fill plan with artifact/receipt paths, after-approval runtime statuses, missing-fill counts, and validator-command coverage, renders detailed per-request queues with validator/build command blocks, and requires candidate trace receipts plus managed/dense receipts to point back to existing saved request and prompt-set files with approvals recorded on that request; `scripts/plan_phase3_dense_fallback_capture.py` breaks the fallback blocker into fill-output, comparison-artifact, updated-bundle requests, and an optional Markdown handoff; `scripts/plan_phase3_policy_candidate_trace.py` breaks the policy-candidate blocker into candidate router trace, ready trace receipt with existing source request, prompt-set, and candidate-trace path bindings, replay validation, updated-bundle requests, and explicit pre-capture planned trace paths, and an optional Markdown handoff; `scripts/build_phase3_live_capability_proof_template.py` builds valid not-ready proof templates with source bundle and prompt context, `scripts/plan_phase3_handoff_coverage.py` validates prompt, trace-receipt, output, request, planned-only launch-card template, and live-proof scaffold coverage across all six bundles and can write a Markdown coverage/repair report, `scripts/plan_phase3_runtime_capture_request.py` confirms all six saved requests are valid, drift-free, and have required validator-command coverage, and `scripts/plan_phase3_runtime_actuator_design.py` validates the selected backend actuator design boundary and explicit residency/control/cleanup gaps from the managed-loading plan; `scripts/plan_phase3_runtime_actuator_spike.py` turns those gaps into proof requirements, dependency order, and completion gates for the next llama.cpp actuator spike; `scripts/plan_phase3_live_capability_proof.py` validates future live proof artifacts, requires the source-bundle path to exist, can enforce expected model/backend/prompt/source-bundle context when a proof is attached, requires exported artifact paths to exist once artifact export is marked available, and can write a Markdown handoff for residency/control plus cleanup proof blockers.
- The artifact intake planner scans generated router traces and optional inventory summaries, can emit Docker/Ollama-volume scanner commands for traced GGUF blobs, then reports whether each candidate is handoff-ready or still needs a scanner-derived manifest and fallback comparison.
- Real-evidence bundle validation binds those artifacts, policy-candidate trace receipts, and approval metadata into valid-but-not-ready or ready-for-Phase-4 handoff manifests; attached live proof must match the bundle model, backend, prompt family, and source bundle path, ready live-proof exports must point to existing files, and invalid bundle-local packet evidence is surfaced as a bundle validation error. The repo-local matrix currently has six real trace/inventory/replay bundles and remains not ready because no bundle has a replay policy candidate, fallback quality, or a populated live capability proof artifact; it now reports policy-candidate blocker counts, trace-receipt required/ready/blocked counts, and no-reuse-distance and prompt-identity diagnostics. The top-level go/no-go decision can emit a Markdown decision report and consumes that matrix, handoff scaffold coverage, runtime-capture request audit, and the matrix-derived policy-candidate trace receipt gate, capture-result intake receipt gate, and receipt-fill manifest so validated scanner-derived pairing is not reported as missing, policy-candidate blockers expose blocked-bundle, no-reuse-distance, and prompt-identity counts plus the next candidate-trace operator step, launch-card template/binding/runtime-command counts stay visible, runtime-actuator spike proof counts stay visible, unfilled or drifted capture artifacts stay visible, and the evidence packet emits an ordered promotion checklist, runtime-capture queue selection, the approval metadata rebuild command manifest, the approval-manifest parity check, the receipt-requirement parity check, the path-aware operator-queue parity, recommended and all-request post-approval capture-fill parity checks, the all-request capture queue manifest, the all-request approval command manifest, the recommended runtime-capture sequence, the recommended post-approval preview, the recommended runtime-capture preflight manifest, the recommended runtime-capture command contract, the recommended and all-request runtime-capture execution coverage manifests, recommended launch-card template export, metadata-only operator handoff package export and validation with a reuse-evidence capture plan artifact, the repo-level launch-card library and prompt-set-bound binding handoff summary, the recommended filled launch-card binding intake, repo-wide filled launch-card directory intake, reuse-evidence capture plan handoff validation, the recommended and all-request runtime-capture runtime-command/manual execution coverage, the all-request manual capture runbook, the all-request post-capture intake runbook, the recommended post-approval capture-fill plan, the recommended policy-candidate trace handoff, the recommended dense fallback capture handoff, the recommended live capability proof handoff, the recommended and all-request validator command manifests, the all-request downstream handoff manifest, the all-request receipt-fill manifest, the all-request receipt-fill command manifest, the Phase 3 blocker closure manifest with mapping-ready/evidence-complete ready-scope status, Phase 3 blocker evidence ledger with row-level missing-evidence counts, Phase 3 blocker resolution queue with ordered work packages, next-unblocked work-package pointer, dependency order, completion gates, embedded runtime-actuator proof handoff, and closure-level validator counts, and the all-request receipt/validator parity check, receipt-fill artifact-class count parity, receipt-command validator coverage, validator-manifest runtime/future coverage, separating satisfied matrix/scaffold gates from approval-bound capture, intake, fallback-quality, runtime-actuator design, runtime-actuator spike proof handoff, live-proof, and Phase 4 promotion blockers. Bundle-local evidence packets keep their decisions bundle-local, and a policy-candidate trace cannot promote without a matching ready trace receipt.
- A positive-path test proves that generated non-fixture trace, scanner-style inventory, dense fallback comparison, and bundle artifacts can promote the packet and capture plan to Phase 4-ready while still refusing to claim live paging.
- Baseline policy definitions validate offline and replay artifacts expose policy-candidate diagnostics, including `no_replay_policy_candidate` when traces have no reuse-distance observations; replay-good candidate traces still need prompt identity metadata on captured events plus a ready trace capture receipt with existing source request, prompt-set, and candidate-trace path bindings before they can promote a bundle.

### Phase 4: Managed Expert Loading Backend Adapter Feasibility Spike

Goal: decide whether a backend adapter can support guarded live managed loading
with residency read/write, fallback, artifacts, and cleanup. `llama.cpp` remains
the first likely substrate, but the same proof applies to
`vllm_openai_compatible` extensions, `hookable_pytorch` runtimes,
`moe_infinity_style` adapters, and `prototype_offload_system` prototypes.

Deliverables:

- A spike document or branch plan identifying exact backend control points.
- Proof requirements for routing visibility, tensor residency control, dense
  fallback, artifact shape, cleanup, and rollback.
- A comparison between a controller outside the runtime, a small `llama.cpp`
  patch or fork, a backend plugin, and a hookable runtime adapter.

Evidence gate:

- There is a concrete runtime hook, adapter, or patch point for residency read,
  pin, evict, preload, demote, or expert offload behavior.
- The patch can produce auditable artifacts without breaking dense baseline
  execution.
- Cleanup and fallback are tested under failure.

### Phase 5: Guarded Runtime Actuator Experiment

Goal: run a narrowly scoped live actuator experiment only after the prior gates
have passed.

Deliverables:

- A guarded actuator mode that can preload, pin, evict, or restrict expert
  residency in a live runtime.
- Kill switch, dense fallback, artifact export, and cleanup are mandatory.
- Comparisons against baseline and replay expectations are automatic.

Evidence gate:

- Actuator behavior is externally observable and internally auditable.
- Dense fallback remains available for every run.
- The experiment demonstrates a measurable memory or latency effect without
  unacceptable quality regression.

## Decision Gates

Do not advance from a phase until its evidence gate is satisfied by artifacts,
not by intent.

- Phase 0 to Phase 1: manifest, roadmap, managed-loading contract, and planner
  checks pass locally.
- Phase 1 to Phase 2: passive external observation surfaces and limitations are
  explicit; live baseline artifact capture can continue in parallel while
  hookable tracing advances.
- Phase 2 to Phase 3: semantic traces exist from a hookable runtime and the
  repeat protocol can validate trace artifacts without rerunning live models.
- Phase 3 to Phase 4: inventory manifests, dry-run layout planning, replay
  metrics, rejected-policy evidence, dense fallback comparison, validated real-evidence bundles, capture-result intake receipt-gate evidence, top-level go/no-go evidence, and selected runtime-capture preflight, policy-candidate trace receipt and prompt-identity/reuse evidence, validated filled-launch-card binding consumed by the evidence packet, and execution-coverage evidence exist. A
  live adapter is not considered until the Phase 3 decision says the policy deserves a
  live spike.
- Phase 4 to Phase 5: a backend adapter or runtime actuator point exists, can
  read/write residency state, and has a rollback plan.

## llama.cpp Controller And Fork Options

A controller outside `llama.cpp` can schedule runs, choose probe suites, inspect
Model Plane manifests, analyze artifacts, and recommend expert residency. That
is useful, but it cannot by itself load only routed experts.

Real expert paging needs a runtime actuator, hook, or patch that can pin, evict,
preload, or otherwise alter expert tensor residency or expert offload behavior.
If `llama.cpp` remains the right substrate, prefer the smallest controller
patch or fork that exposes those controls and artifacts. A large custom runtime
is a last resort, not the starting point.

The actuator feasibility spike must prove:

- routing visibility: routed expert ids or sufficient router metadata are
  available at the decision point
- residency control: expert tensors can be pinned, evicted, preloaded, or
  offloaded according to policy
- dense fallback: normal dense/full-residency behavior remains available
- artifact shape: decisions and outcomes can be emitted in
  `memory-moe-bridge-v1`-compatible form
- cleanup: failed or interrupted runs restore runtime state
- isolation: experiments do not require global machine or model-cache mutation

## Model Plane Cron And Callable-Function Handoff

Model Plane owns orchestration surfaces such as callable cron functions, profile
validation, launch/health/log inspection, and run-scoped MoE manifest export.
MoE Run Anyway should consume those manifests and produce a planning artifact,
not reach back into Model Plane internals.

The planned handoff is:

```text
Model Plane callable function or cron -> run-scoped MoE manifest
  -> MoE manifest planner -> planned harness_run_request
  -> user-approved probe or replay command
```

`harness_run_request` is a planning stage, not an execution API yet. It should
name the target class, approved command class, expected artifacts, safety
contract, and missing actuator capabilities.

## Risks

- Timing and endpoint metrics may be overinterpreted as semantic routing.
- A runtime patch may expose routing but still lack safe tensor residency
  control.
- A controller policy can look good in replay while causing live churn or
  cleanup failures.
- Baseline artifacts from different backends may not be comparable without
  strict labels and shared fields.
- Fork maintenance cost can exceed the benefit unless the actuator patch stays
  small.

## Non-Goals

- No claim of live expert paging before an actuator exists.
- No model downloads, token use, Docker runs, model server launch, or prompt
  traffic from roadmap/planner checks.
- No custom runtime rewrite as the first actuator plan.
- No semantic expert-id inference from stock OpenAI-compatible endpoint
  telemetry.
- No destructive cleanup or mutation of user model caches.
