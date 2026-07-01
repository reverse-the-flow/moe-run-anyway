# Flash-MoE Inspiration For The Next Step

This note summarizes architecture and workflow lessons from the local
`references/inspiration-code/checkouts/flash-moe` checkout. It is for design
inspiration only. Do not copy implementation code or project text into MoE Run
Anyway unless licensing and attribution are clarified.

## Useful Patterns

### Expert Index First

Flash-MoE uses an expert index before attempting on-demand expert loading. The
useful shape is layer-oriented and records source file, byte offset, expert
stride, expert byte size, tensor shape, and component name.

MoE Run Anyway should make this a first-class offline artifact:

- `model_id`
- `source_format`: `safetensors`, `gguf`, or another explicit value
- `backend_family`
- `layer_id`
- `expert_id`
- `component_name`
- `source_file`
- `byte_offset`
- `byte_length`
- `stride_bytes`
- `dtype`
- `shape`
- `estimated_residency_bytes`
- `coverage_status`

This should be generated without loading tensor values into RAM.

### Dry-Run Packing Before Writing Stores

Flash-MoE repacks scattered expert tensors into fixed per-layer expert files.
That is a useful future actuator substrate, but MoE Run Anyway should not jump
straight to writing packed stores.

The safe next step is a streaming packer dry run:

- read the expert inventory manifest
- compute a proposed output layout
- estimate disk required
- report missing components
- write no tensor data unless an explicit write mode is approved

### Measurement Before Clever Caches

Flash-MoE is valuable because it records negative results. Several plausible
ideas lost in full-pipeline tests: custom caches, compression, prediction,
prefetch hints, `mmap` expert files, and alternative async I/O paths.

For MoE Run Anyway, that means replay should answer these questions before any
live actuator work:

- How many unique experts are routed per prompt family?
- What is the reuse distance for `(layer_id, expert_id)`?
- How many bytes would a policy request?
- What is the miss rate under a fixed resident budget?
- How much churn does the policy create?
- Which policies are rejected, and why?

### Keep Evidence Labels Separate

Flash-MoE separates routing, expert I/O, and compute timing. MoE Run Anyway
should keep equivalent labels separate:

- `routing_trace`: selected experts and weights
- `inventory`: source layout and estimated bytes
- `policy_replay`: keep, preload, evict, fallback decisions
- `runtime_baseline`: endpoint timing and observability
- `live_actuator`: future before/after residency and cleanup proof

That prevents stock endpoint timing from being mistaken for semantic routing or
managed expert loading.

## Recommended Next Step

Build the offline expert inventory manifest path.

This is the best bridge between current Phase 2 router traces and later managed
loading. It does not require a live actuator, GPU work, Docker, or a full model
load into RAM.

Proposed sequence:

1. Add `data/expert_inventory_manifest.fixture.json`.
2. Add `scripts/plan_expert_inventory.py` as a dependency-free validator and
   summarizer.
3. Add tests for missing layers, missing components, duplicate expert entries,
   and byte-size totals.
4. Add a safetensors header/index scanner dry run after the schema is stable.
5. Add a GGUF tensor-table scanner path after the safetensors path is proven.
6. Feed router traces plus inventory into Phase 3 replay metrics: unique
   experts, estimated bytes, reuse distance, miss rate, churn, and fallback
   frequency.

## What Not To Do Yet

- Do not build a custom expert cache before replay says a cache policy is worth
  testing.
- Do not write packed expert stores before the dry-run manifest can prove
  layout, disk cost, and missing tensors.
- Do not claim stock Ollama or stock endpoint telemetry exposes semantic
  routing.
- Do not claim live expert paging until residency observation, residency
  control, dense fallback, cleanup, and artifact export exist.

## Launch Card Implication

The patched llama.cpp trace launch card should eventually reference both:

- the semantic trace gate: expected layers and required router tensor kinds
- the inventory gate: expected expert count, top-k, routed layer count, and
  estimated expert bytes

That gives agents a concrete way to decide whether a run is ready for replay,
still just a trace, or blocked before actuator work.
