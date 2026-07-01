# Phase 3 Real Evidence

This folder holds repo-local Phase 3 evidence artifacts that are not fixtures.
They are offline audit artifacts only, not live expert paging proof.

## Current Matrix

Run the matrix from the repository root:

```bash
uv run --managed-python --python 3.13 scripts/plan_phase3_real_evidence_matrix.py
uv run --managed-python --python 3.13 scripts/plan_phase3_handoff_coverage.py
uv run --managed-python --python 3.13 scripts/plan_phase3_live_capability_proof.py
uv run --managed-python --python 3.13 scripts/plan_phase3_launch_card_library.py
```

Current matrix facts:

- 6 real-model trace/inventory pairs validate and replay: PC Mixtral, PC Qwen3 Coder A3B, PC Qwen3 Coder 30B alias, PC Nemotron Cascade, PC Nemotron Nano A3B, and PC Gemma4 26B A4B.
- 0 bundles are Phase 4-ready.
- Remaining common blockers are no replay policy candidate, dense/full-runtime fallback comparison, live residency observation/control, and cleanup/restore proof.
- 0 bundles are policy-candidate-ready: the current real traces replay cleanly, but they do not yet contain reuse-distance observations that make a managed loading policy worth trying.
- Each real bundle now has a shared 8-prompt repeated prompt set for policy-candidate and dense fallback capture.
- Each real bundle now has managed and dense saved-output templates with all prompt ids covered and outputs still missing.
- The dense fallback comparison now has bundle-specific capture scaffolds; all prompt sets and output-summary templates are scaffold-ready, while actual managed/dense runtime outputs and ready capture receipts with existing source request and prompt-set bindings remain approval-bound.
- The policy-candidate blocker now has bundle-specific trace scaffolds; fillable trace receipt templates are scaffold-ready, while candidate router traces, ready trace receipts with existing source request, prompt-set, and candidate-trace path bindings, and replay validation remain approval-bound.
- Each real bundle now has a runtime-capture request that binds approval-bound runtime fills, trace-receipt templates, and the future live-proof fill into one request with validator commands.
- Each real bundle now has a planned-only runtime-capture launch-card template beside that request; the templates are structurally valid and non-executable, with real Model Plane command bindings still missing by design.
- The repo also keeps a saved launch-card binding worksheet and a saved Model Plane artifact-writer contract request for the whole six-bundle library; the launch-card library planner audits both for missing files or drift and refreshes them with `--write-default-handoff-artifacts`.
- Each real bundle now has a valid not-ready live-capability proof template for future residency observation/control and cleanup/restore evidence.
- 0 scanner coverage-gap inventories remain after the GGUF scanner learned Nemotron-H `up/down` and Gemma4 fused `gate_up` component profiles.

## Valid Bundles

| Bundle | Trace | Inventory | Joined routes | Unique experts | Status |
| --- | --- | --- | ---: | ---: | --- |
| `pc_gemma4_26b_a4b_phase3_real_evidence_bundle.json` | `pc_gemma4_26b_a4b_router_events.jsonl` | `sha256-2f8672b0c2cca8dedfb8782815c2769ccdaa6512788f3ee87b32cf117f0dffc1.expert_inventory.json` | 960 | 960 | valid, not Phase 4-ready |
| `pc_mixtral_phase3_real_evidence_bundle.json` | `pc_mixtral_router_events.jsonl` | `sha256-5041ba4278429fe475782b889471b5ff065a6cce5c3a539bd61d1e457f1961de.expert_inventory.json` | 250 | 250 | valid, not Phase 4-ready |
| `pc_nemotron_cascade_phase3_real_evidence_bundle.json` | `pc_nemotron_cascade_router_events.jsonl` | `sha256-916a371ed63edd6602da950df2c1eed5ff74bfb9dac2ece64c7cabaafb3e2922.expert_inventory.json` | 414 | 414 | valid, not Phase 4-ready |
| `pc_nemotron_nano_a3b_phase3_real_evidence_bundle.json` | `pc_nemotron_nano_a3b_router_events.jsonl` | `sha256-a70437c41b3b0b768c48737e15f8160c90f13dc963f5226aabb3a160f708d1ce.expert_inventory.json` | 414 | 414 | valid, not Phase 4-ready |
| `pc_qwen3_coder_a3b_phase3_real_evidence_bundle.json` | `pc_qwen3_coder_a3b_router_events.jsonl` | `sha256-17d51f5310e9a598e5ac914d30f401fb2d1bc3b6a06a846919099eac09364ae1.expert_inventory.json` | 760 | 760 | valid, not Phase 4-ready |
| `pc_qwen3_coder_30b_alias_phase3_real_evidence_bundle.json` | `pc_qwen3_coder_30b_alias_router_events.jsonl` | `sha256-1194192cf2a187eb02722edcc3f77b11d21f537048ce04b67ccf8ba78863006a.expert_inventory.json` | 760 | 760 | valid, not Phase 4-ready |

All six inventories were generated metadata-only from the read-only Docker
Ollama volume. The scanner read GGUF metadata and tensor tables only; it did
not load tensor values, write packed stores, launch model servers, or send
prompt traffic.

## Scanner Component Profiles

The GGUF scanner now records the component profile it inferred from the tensor
table:

- Mixtral/Qwen-style `gate_up_down`: separate routed `gate_proj`, `up_proj`, and `down_proj` expert tensors.
- Nemotron-H `nemotron_h_up_down`: routed `up_proj` and `down_proj` expert tensors, with router and shared expert tensors left resident-side.
- Gemma4 `gate_up_down`: fused `ffn_gate_up_exps.weight` is split into metadata-only `gate_proj` and `up_proj` slices, with `down_proj` and `down_proj_scale` rows recorded separately.

These profiles make inventory/replay evidence stronger, but they still do not
prove live expert paging. Phase 4 remains blocked until at least one replay
policy candidate, fallback quality, residency observation/control, and
cleanup/restore artifacts exist.

## Live Capability Proof

`scripts/build_phase3_live_capability_proof_template.py` builds fillable proof templates, and `scripts/plan_phase3_live_capability_proof.py` validates future live proof
artifact for residency observation, residency control, cleanup/restore, and
artifact export. With no artifact it records the exact blocker
`live_capability_proof_artifact_missing`; with an artifact it requires before
and after residency state, non-dry-run control actions, verified cleanup, a
failure-path test, and exported proof paths.

## Useful Checks

```bash
uv run --managed-python --python 3.13 scripts/plan_phase3_real_evidence_matrix.py
uv run --managed-python --python 3.13 scripts/plan_phase3_handoff_coverage.py
uv run --managed-python --python 3.13 scripts/plan_phase3_live_capability_proof.py
uv run --managed-python --python 3.13 scripts/plan_phase3_evidence_packet.py
uv run --managed-python --python 3.13 scripts/build_phase3_prompt_set.py memory-moe-mvp/phase3-real-evidence/pc_mixtral_phase3_real_evidence_bundle.json --default-output --json
uv run --managed-python --python 3.13 scripts/build_phase3_output_summary.py memory-moe-mvp/phase3-real-evidence/pc_mixtral_phase3_real_evidence_bundle.prompt-set.json --output-label managed --default-output --json
uv run --managed-python --python 3.13 scripts/build_phase3_output_summary.py memory-moe-mvp/phase3-real-evidence/pc_mixtral_phase3_real_evidence_bundle.prompt-set.json --output-label dense --default-output --json
uv run --managed-python --python 3.13 scripts/build_phase3_runtime_capture_request.py memory-moe-mvp/phase3-real-evidence/pc_mixtral_phase3_real_evidence_bundle.json --default-output --json
uv run --managed-python --python 3.13 scripts/plan_phase3_runtime_capture_commands.py memory-moe-mvp/phase3-real-evidence/pc_mixtral_phase3_real_evidence_bundle.runtime-capture-request.json --output-launch-card memory-moe-mvp/phase3-real-evidence/pc_mixtral_phase3_real_evidence_bundle.runtime-capture-launch-card.template.json --json
uv run --managed-python --python 3.13 scripts/plan_phase3_launch_card_library.py --write-default-handoff-artifacts --json
uv run --managed-python --python 3.13 scripts/build_phase3_live_capability_proof_template.py memory-moe-mvp/phase3-real-evidence/pc_mixtral_phase3_real_evidence_bundle.json --default-output --json
uv run --managed-python --python 3.13 scripts/plan_phase3_dense_fallback_capture.py memory-moe-mvp/phase3-real-evidence/pc_mixtral_phase3_real_evidence_bundle.json
uv run --managed-python --python 3.13 scripts/plan_phase3_policy_candidate_trace.py memory-moe-mvp/phase3-real-evidence/pc_mixtral_phase3_real_evidence_bundle.json
# Add --live-proof-artifact-path path/to/live-proof.json when a proof artifact exists.
uv run --managed-python --python 3.13 scripts/plan_phase3_real_evidence_bundle.py memory-moe-mvp/phase3-real-evidence/pc_mixtral_phase3_real_evidence_bundle.json
uv run --managed-python --python 3.13 scripts/plan_phase3_real_evidence_bundle.py memory-moe-mvp/phase3-real-evidence/pc_nemotron_cascade_phase3_real_evidence_bundle.json
uv run --managed-python --python 3.13 scripts/plan_phase3_real_evidence_bundle.py memory-moe-mvp/phase3-real-evidence/pc_gemma4_26b_a4b_phase3_real_evidence_bundle.json
```