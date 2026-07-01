# Phase 2 Repeat Protocol

Phase 2 is the semantic routing trace phase. Its job is to prove that a
hookable runtime can emit routed expert choices, not to prove expert residency,
paging, preload, eviction, or cleanup.

## Boundary

There are three different evidence levels:

1. Matrix validation: the current inventory and blockers are internally
   consistent.
2. Metadata preflight: a GGUF blob is MoE-shaped and has routed expert tensors
   without loading model weights.
3. Semantic router trace: a patched runtime emits nonempty
   `memory-moe-bridge-v1` JSONL with selected expert ids and selected weights.

Only level 3 may be counted as `semantic_trace_captured`.

Ollama models are valid inputs only when their GGUF blobs are mounted read-only
and run through the patched direct llama.cpp trace binary. Stock Ollama API
traffic is runtime evidence, not semantic router trace evidence.

## Cheap Repeat Gate

Run these from the repository root before any live model attempt:

```bash
python scripts/plan_hookable_moe_attempts.py --json
python scripts/validate_llama_cpp_router_trace.py \
  memory-moe-mvp/data/llama_cpp_router_trace.fixture.jsonl \
  --expected-layers 2 \
  --require-kind selected_experts \
  --require-kind selected_weights \
  --require-kind selected_weights_norm \
  --json
python -m unittest discover -s memory-moe-mvp/tests -p "test*.py"
```

The pass is healthy when the matrix is valid, the fixture trace is valid, and
the unit suite passes.

## GGUF Preflight Gate

Before loading a large model, run the dependency-free GGUF preflight against
the model file or Ollama blob:

```bash
python scripts/gguf_moe_preflight.py /path/to/model.gguf --json
```

This gate is enough to classify model shape, routed layer count, expert count,
top-k, shared experts, and tensor-table compatibility. It does not prove that
the patched runtime can load the model or emit router events.

## Full Trace Gate

A full patched llama.cpp trace attempt should write generated artifacts outside
the committed tree, usually under `.codex_tmp/` or another ignored run folder.

Acceptance criteria:

- process exits successfully, or emits a complete trace before a known late
  process error
- trace JSONL is nonempty
- validator passes with required tensor kinds:
  - `selected_experts`
  - `selected_weights`
  - `selected_weights_norm`
- layer count matches the expected routed layer count
- summary records top-k and unique expert coverage
- `hookable_moe_attempt_matrix.json` is updated only after the trace validates

Timeouts with empty JSONL are runtime budget failures, not semantic trace
failures. Record them as repeat blockers unless the model also has a distinct
load error or tensor compatibility error.

## 2026-06-27 Repeat Check

The cheap repeat gate passed after fixing Windows drive-letter path handling in
`scripts/plan_hook_trace_capture.py`:

- matrix validator: valid, 20 attempts, 8 semantic traces
- trace fixture validator: valid
- unit suite: 97 tests passed

A fresh PC Mixtral full-trace repeat was attempted through
`memory-moe-llama-router-trace:latest` against the read-only Docker Ollama blob.
The attempt exceeded a 15-minute guardrail and left an empty
`router-events.jsonl`, so it did not refresh semantic evidence. Treat this as a
repeat-window/runtime-budget issue. The prior validated Mixtral semantic trace
remains the current evidence until a longer or cleaner run window completes.
