# Model Plane Manifest Consumption

MoE Run Anyway consumes the Model Plane MoE probe manifest as the durable bridge
from local runtime orchestration into probe planning. The manifest should come
from Model Plane's backend API or callable functions, not from values copied out
of the human console.

The bridge flow is:

```text
Model Plane callable function or cron -> validate/launch/health/logs
  -> run-scoped MoE probe manifest -> planner
  -> planned harness_run_request -> approved probe path
```

The human UI is useful for status and inspection. The machine-readable manifest
is the contract agents should pass between repos.

Model Plane owns cron/callable orchestration, profile validation, launch/health
checks, log discovery, and run-scoped manifest export. MoE Run Anyway owns
manifest consumption, safe probe planning, artifact contracts, and controller
replay. The planner may describe a `harness_run_request`, but that request is a
planning artifact, not a runtime executor.

## Planner Entry Point

From the repository root:

```bash
python3 scripts/plan_moe_probe_manifest.py /path/to/moe-probe-manifest.json
```

Add `--json` when another agent or script needs the plan as structured output:

```bash
python3 scripts/plan_moe_probe_manifest.py /path/to/moe-probe-manifest.json --json
```

The repository includes a saved runtime-baseline handoff fixture for local
roadmap validation:

```bash
python3 scripts/plan_moe_probe_manifest.py memory-moe-mvp/data/model_plane_moe_probe_manifest.runtime_baseline.fixture.json --json
```

The planner validates only the bridge contract. It does not start model servers,
download models, authenticate, inspect private tokens, run Docker, or send
prompt traffic.

Structured output includes a `planned_harness_run_request` object with:

- `stage: harness_run_request`
- `status: planned_only`
- selected `target_class`
- safe command class and deferred live command class
- expected artifact class
- missing actuator capabilities such as expert pin, evict, preload, and
  load-only-routed-experts

This planned stage exists so Model Plane agents and MoE agents can agree on the
next harness step without implying that a live expert-paging actuator exists.

## Required Manifest Fields

The planner accepts `schema_version: model-plane-moe-probe-manifest-v1` and
requires these non-empty fields:

- `profile_id`
- `model_id`
- `backend_family`
- `base_url`
- `primary_probe_hint`

It also consumes:

- `health_url`
- `container_name`
- `log_file_path`
- `model_path`
- `semantic_expert_ids`
- `hookable_runtime_available`
- `passive_sidecar_requested`
- `runtime_observability.expected_paths`
- `safety_notes`

`semantic_expert_ids` must be one of:

- `not_exposed`
- `expected_when_router_outputs_are_exposed`
- `runtime_dependent`

## Decision Policy

Agents should let `scripts/plan_moe_probe_manifest.py` make the final path
selection, but the policy is intentionally simple:

| Manifest signal | Selected target class | Meaning |
| --- | --- | --- |
| `primary_probe_hint=runtime_baseline` with `backend_family=llama_cpp` or unknown safe default | `stock_llama_cpp_openai_compatible` | Use guarded llama.cpp runtime baseline planning against the existing endpoint. |
| `backend_family=vllm_openai_compatible`, `ollama_openai_compatible`, or `openai_compatible` | `openai_compatible_runtime` | Use guarded OpenAI-compatible runtime planning where `/v1/models` or `/models` can satisfy readiness. |
| `primary_probe_hint=passive_sidecar` | `passive_sidecar_proxy` | Put the passive sidecar between client traffic and the upstream endpoint. |
| `primary_probe_hint=hookable_pytorch` and `hookable_runtime_available=true` | `hookable_pytorch_moe` | Use the hookable semantic path where router outputs may be captured. |
| Hookable hint without `hookable_runtime_available=true` | invalid manifest | Refuse to plan semantic probing. |

Runtime baseline plans emit only `run_live_baseline.py --dry-run` and
`run_live_baseline.py --preflight-only` commands. These are planning and
readiness commands, not prompt-traffic runs. Runtime baseline commands pass the
manifest `backend_family` through to the runner and runtime probe so artifacts
do not claim `llama_cpp` when the profile is vLLM, Ollama, or another
OpenAI-compatible backend.

Passive sidecar plans emit the sidecar command because the sidecar itself is the
non-invasive observation layer. A separate client must still choose to send
traffic through it.

Hookable PyTorch plans emit the no-model synthetic hook smoke command as a safe
first step. A real local hookable runner remains deferred unless a compatible
checkpoint and runtime already exist.

## Semantic Honesty

Do not infer semantic expert ids from stock endpoint telemetry. For
`llama.cpp`, vLLM, Ollama, and generic OpenAI-compatible endpoints, Model Plane
can provide base URLs, health URLs, log paths, and observability hints; MoE Run
Anyway can collect runtime/request evidence from those surfaces. That evidence
is useful, but it is not a router trace.

Semantic expert ids require one of:

- a hookable PyTorch/Transformers-style runtime exposing router modules or
  router outputs
- a future backend patch that emits semantic expert ids
- a model-specific runner that writes compatible `memory-moe-bridge-v1`
  artifacts

The manifest field `semantic_expert_ids` is therefore a guardrail, not a claim
of coverage.

## Agent Handoff Shape

A safe agent loop is:

1. Ask Model Plane for `GET /profiles`.
2. Choose a profile and call `POST /profiles/{profile_id}/validate`.
3. Use Model Plane callable launch, health, log-inspection, or cron functions
   only when the user has approved runtime actions.
4. Fetch or receive the run-scoped MoE probe manifest, such as
   `GET /profiles/{profile_id}/moe-probe-manifest` or a callable-function
   export artifact.
5. Save the JSON manifest.
6. Run `python3 scripts/plan_moe_probe_manifest.py manifest.json --json`.
7. Review the returned `planned_harness_run_request`.
8. Present or execute only the returned command class the user has approved.

This keeps Model Plane as the orchestration/control layer and MoE Run Anyway as
the probe planner and harness. The user should not have to copy endpoints,
ports, model ids, or log paths between applications.

## Phase 3 Runtime Capture Cards

For Phase 3 runtime-capture work, `scripts/plan_phase3_runtime_capture_commands.py --output-launch-card path/to/card.json` exports a planned-only launch-card template from the selected saved request. The top-level packet can also export the recommended card with `scripts/plan_phase3_evidence_packet.py --output-runtime-capture-launch-card path/to/card.json`. The template can carry Model Plane profile context, prompt-set, artifact, receipt, request, approval-gate, binding-template, and command-placeholder metadata, but it is not executable until Model Plane attaches real callable ids or launch commands after approval.

For the whole repo, `scripts/plan_phase3_launch_card_library.py --output-binding-worksheet path/to/worksheet.json` writes the 18-task worksheet Model Plane can fill. `scripts/plan_phase3_launch_card_library.py --output-model-plane-contract-request path/to/request.json` writes a metadata-only artifact-writer contract request with the exact Phase 3 writer descriptors and per-task request, prompt-set, artifact, and receipt paths Model Plane needs under `card.phase3_artifact_writers[]`. The default saved copies live in `memory-moe-mvp/phase3-real-evidence/phase3-launch-card-binding-worksheet.json` and `memory-moe-mvp/phase3-real-evidence/phase3-model-plane-artifact-writer-contract-request.json`; the library planner audits them for missing files or drift, and `--write-default-handoff-artifacts` refreshes both without running runtimes. After callable ids or launch commands are filled, `scripts/plan_phase3_launch_card_library.py --binding-worksheet path/to/worksheet.json --output-filled-card-dir path/to/cards --json` validates the worksheet offline and emits filled launch cards without running them. If Model Plane returns a contract fulfillment instead of a worksheet, `scripts/plan_phase3_launch_card_library.py --model-plane-contract-fulfillment path/to/fulfillment.json --output-filled-card-dir path/to/cards --json` validates schema, exact task identity, request/prompt-set/artifact/receipt paths, approval gates, prompt-traffic acknowledgement, and command readiness before writing the same filled-card format.

For a single operator/agent folder, `scripts/plan_phase3_evidence_packet.py --output-operator-handoff-dir path/to/handoff` writes the evidence packet report, handoff README, binding worksheet, Model Plane artifact-writer contract request, recommended planned launch-card template, recommended runtime-capture preflight, recommended runtime-capture command contract, recommended runtime-capture execution coverage, all-request runtime-capture execution coverage, runtime-capture launch-card directory manifest, reuse-evidence capture plan, manual capture runbook, post-capture intake runbook, capture queue, approval command manifest, validator manifests, downstream handoff manifest, receipt-fill manifests, blocker-closure manifest, blocker-evidence ledger, and blocker-resolution queue without running the runtime. `scripts/plan_phase3_operator_handoff.py path/to/handoff` then validates required files, schemas, safety flags, recommended and all-request approval-command readiness, approval-command-manifest/capture-queue parity, recommended capture-queue agreement, recommended launch-card worksheet/template/capture-queue task parity, Model Plane artifact-writer contract request/schema/task parity, selected preflight packet/file parity, selected command-contract packet/file parity, selected execution-coverage packet/file parity, all-request execution-coverage packet/file parity, runtime-capture launch-card directory packet/execution/handoff parity with external staging directories allowed but repo-relative request and source-card paths checked, reuse-evidence capture-plan packet/gap/handoff/recommended-capture parity, manual-capture runbook packet/file parity, post-capture intake runbook packet/file parity, post-capture artifact-gate/manual-task/receipt-command parity, manual-task/receipt-command parity, all-request binding-worksheet/capture-queue task parity, repo-relative artifact/request/receipt path safety, counts, receipt-command/capture-queue artifact-key parity, receipt-command validator coverage against capture-queue validator counts, validator-manifest runtime/future coverage, downstream handoff request/section/path coverage, blocker-closure reason/action/path coverage, blocker-evidence ledger row/count/packet parity, blocker-resolution queue package/row/dependency/completion-gate/validator parity plus runtime-actuator proof-handoff counts, requirement coverage, and control-dependency checks, and README markers offline.

After Model Plane fills a card directly, or after the library planner emits filled cards from a worksheet, rerun `scripts/plan_phase3_runtime_capture_commands.py --launch-card path/to/filled-card.json` against the same request. The intake remains offline and only marks runtime-command binding ready when every capture task has matching artifact, receipt, request, and prompt-set binding paths, explicit approval gates, and either a callable id or a real launch command. Unfilled templates remain valid but blocked as `runtime_command_binding_missing`; path or schema drift is invalid. The top-level evidence packet accepts one filled card with `scripts/plan_phase3_evidence_packet.py --runtime-capture-launch-card path/to/filled-card.json` for the recommended request, or a repo-wide filled-card directory with `--runtime-capture-launch-card-dir path/to/cards` for all queued requests, writes that directory manifest into `runtime-capture-launch-card-directory.json` inside the operator handoff package, and lets execution coverage move from manual-only to command-backed without launching anything during validation.

The repo-local Phase 3 handoff coverage planner now treats the saved planned-only launch-card template as its own scaffold artifact beside each runtime-capture request. That makes Model Plane binding readiness visible before runtime traffic: missing templates are repairable scaffold gaps, valid unfilled templates are handoff-ready but command-unbound, and filled cards are still validated offline before they can affect execution coverage.

## Expert Paging Boundary

The current bridge can plan runtime baselines, passive sidecar capture,
hookable PyTorch semantic traces, and controller replay. It still cannot page
experts. Real expert paging needs a future runtime actuator, hook, or patch
that can control expert tensor residency or expert offload behavior.

For `llama.cpp`, an external controller can schedule runs, select probe suites,
consume artifacts, and recommend residency. Loading only routed experts requires
a `llama.cpp` hook, controller patch, or fork that can prove routing visibility,
tensor residency control, dense fallback, compatible artifacts, and cleanup.
Those proof requirements are tracked in
[expert-paging-roadmap.md](expert-paging-roadmap.md) and
[../data/expert_paging_roadmap.json](../data/expert_paging_roadmap.json).
