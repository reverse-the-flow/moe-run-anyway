# llama.cpp Engine Hook Track

## Correction

The project should treat llama.cpp MoE models as hook candidates.

The earlier distinction was too narrow: stock GGUF/Ollama targets are not
Python `register_forward_hook()` targets, but the MoE routing decision exists
inside the inference engine when llama.cpp runs the model. Semantic route traces
therefore require an engine-level hook, not a Python forward hook.

## Current Stock Surface

The current llama.cpp image is useful but not sufficient:

- `llama-server` exposes `/metrics`, `/slots`, `/props`, and timing/log data.
- `llama-server --help` exposes MoE placement knobs:
  - `--cpu-moe`
  - `--n-cpu-moe`
- the image includes a `llama-eval-callback` binary, which is relevant as an
  engine callback example.

These surfaces do not currently emit selected expert ids, router logits,
router probabilities, or per-token/per-layer expert choices.

## Required Trace Artifact

A llama.cpp engine hook should emit JSONL compatible with the existing
`memory-moe-bridge-v1` semantic trace boundary. Minimum event fields:

- `event_type: llama_cpp_moe_router_event`
- `contract_version: memory-moe-bridge-v1`
- `timestamp`
- `backend_family: llama_cpp`
- `model`
- `layer_id`
- `sequence_id` or slot id when available
- `token_index` or decode step when available
- `selected_experts`
- `selected_scores` or `selected_probs` when available
- `top_k`
- `source: llama_cpp_engine_hook`
- engine build/version metadata

The hook should also write a manifest and summary:

- hook enabled flag and trace file path
- llama.cpp build/version
- model path and GGUF metadata summary
- router event count
- unique expert ids seen
- by-layer expert counts
- known limitations

## Patch Point

The target patch point is after the MoE router/gate has selected top-k experts
and before expert FFN dispatch for the token/window. That is where the engine
should have layer id, expert ids, and routing scores close together.

The patch should be behind an explicit flag, for example:

```text
--moe-router-trace-file /path/to/router-events.jsonl
```

or equivalent environment variable:

```text
LLAMA_MOE_ROUTER_TRACE_FILE=/path/to/router-events.jsonl
```

No trace should be emitted unless explicitly enabled.

## First Substrate

Use direct llama.cpp before Ollama:

1. Patch/build direct llama.cpp.
2. Run `dolphin-mixtral-8x7b.gguf` first because Mixtral has small, known
   routing shape: 8 experts, top 2.
3. Run `qwen3-30b.gguf` second for a larger Qwen3 MoE shape.
4. Treat PC Ollama models as later targets unless they can be run through the
   patched direct llama.cpp binary or a custom Ollama build.

## Current GX10 Result

GX10 now has a direct llama.cpp source checkout at:

```text
/home/codexlab/src/llama.cpp-moe-hook
```

The first patch artifact is:

```text
memory-moe-mvp/patches/llama-cpp-moe-router-trace-example.patch
```

That patch adds a `llama-moe-router-trace` example using llama.cpp's eval
callback. It writes selected expert tensors and selected weight tensors to
JSONL through `LLAMA_MOE_ROUTER_TRACE_FILE`.

Two GX10 direct GGUF runs succeeded:

- Mixtral: 96 events across 32 layers.
- Qwen3: 144 events across 48 layers.

See `llama-cpp-engine-hook-traces-2026-06-25.md`.

## Remaining Blockers

- The JSONL needs a repo validator against the shared trace contract.
- PC Ollama is not instrumented yet; direct patched llama.cpp or a custom Ollama
  build is still needed there.
- Nemotron Super GGUF still has the previous tensor-layout/runtime compatibility
  issue.
- Routing visibility is not residency observation or residency control.
