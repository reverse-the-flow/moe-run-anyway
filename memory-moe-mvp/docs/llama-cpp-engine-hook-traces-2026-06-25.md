# llama.cpp Engine Hook Traces - 2026-06-25

## Result

Patched direct llama.cpp produced real MoE router trace artifacts for two GX10
GGUF models:

- `dolphin-mixtral-8x7b.gguf`
- `qwen3-30b.gguf`

This is semantic route evidence, not passive endpoint timing. It captures the
selected expert ids and selected expert weights emitted by the llama.cpp graph.

## Patch

The patch artifact is:

```text
memory-moe-mvp/patches/llama-cpp-moe-router-trace-example.patch
```

It adds a `llama-moe-router-trace` example to a llama.cpp checkout. The example
sets `params.cb_eval` and asks the scheduler to retrieve only these tensors:

- `ffn_moe_topk-*`
- `ffn_moe_weights-*`
- `ffn_moe_weights_norm-*`

It writes JSONL to `LLAMA_MOE_ROUTER_TRACE_FILE`. The event shape uses:

- `event_type: llama_cpp_moe_router_tensor`
- `contract_version: memory-moe-bridge-v1`
- `backend_family: llama_cpp`
- `source: llama_cpp_eval_callback`
- model path
- tensor name and kind
- layer id
- GGML type
- tensor shape
- selected values

## GX10 Source Checkout

The working source checkout used for the first run was:

```text
/home/codexlab/src/llama.cpp-moe-hook
```

The upstream commit was:

```text
beac530
```

The build target was:

```text
build-moe-trace/bin/llama-moe-router-trace
```

## Mixtral Trace

Run directory:

```text
memory-moe-mvp/llama-cpp-hook-runs/20260625-mixtral-moe-router-trace-clean
```

Result:

- exit code: 0
- input tokens: 4
- JSONL events: 96
- layers: 0..31
- validator: passed with `scripts/validate_llama_cpp_router_trace.py`
- event counts:
  - `selected_experts`: 32
  - `selected_weights`: 32
  - `selected_weights_norm`: 32
- first top-k tensor shape: `[2,4,1,1]`
- first selected expert values: `[1,0,2,7,4,5,3,6]`
- unique experts seen: 8

## Qwen3 Trace

Run directory:

```text
memory-moe-mvp/llama-cpp-hook-runs/20260625-qwen3-moe-router-trace-clean
```

Result:

- exit code: 0
- input tokens: 2
- JSONL events: 144
- layers: 0..47
- validator: passed with `scripts/validate_llama_cpp_router_trace.py`
- event counts:
  - `selected_experts`: 48
  - `selected_weights`: 48
  - `selected_weights_norm`: 48
- first top-k tensor shape: `[8,2,1,1]`
- first selected expert values:
  `[23,124,12,105,6,64,122,125,75,66,36,42,3,30,116,55]`
- unique experts seen: 126

## Boundary

This proves routing visibility for patched direct llama.cpp. It does not prove
expert residency, expert paging, preload/evict control, or cleanup.

The raw trace directories are generated run artifacts and are ignored by git.

PC Ollama blob access was validated later with the same patched direct llama.cpp
path. See `pc-llama-cpp-router-traces-2026-06-26.md`.

## Next

1. Convert the patch/run command into a repeatable Model Plane launch card.
2. Attempt remaining large PC Ollama MoE candidates through direct patched
   llama.cpp when disk and Docker memory permit.
3. Keep Nemotron HF on the dependency-specific Python hook lane.
