# PC llama.cpp Router Traces - 2026-06-26

## Result

The PC Docker Ollama GGUF models are readable by a patched direct llama.cpp
runner, not only by stock Ollama. Three PC MoE blobs produced validated
`memory-moe-bridge-v1` router trace JSONL through:

```text
memory-moe-llama-router-trace:latest
```

The image is built from:

```text
memory-moe-mvp/docker/llama-moe-router-trace.Dockerfile
```

It packages the `llama-moe-router-trace` example from:

```text
memory-moe-mvp/patches/llama-cpp-moe-router-trace-example.patch
```

## Runtime Shape

Each run mounted the Docker Ollama volume read-only:

```text
open-webui_ollama:/root/.ollama:ro
```

Each run set:

```text
LLAMA_MOE_ROUTER_TRACE_FILE=/out/router-events.jsonl
LLAMA_MOE_ROUTER_TRACE_MAX_EVENTS=512
```

and used CPU-only model placement:

```text
-c 64 -b 4 -ub 4 -ngl 0
```

The image is CPU-only, so llama.cpp prints a GPU warning when `-ngl 0` is
present. That warning is expected and does not block the hook artifact.

## Mixtral

Model:

```text
dolphin-mixtral:8x7b
```

Ollama blob:

```text
/root/.ollama/models/blobs/sha256-5041ba4278429fe475782b889471b5ff065a6cce5c3a539bd61d1e457f1961de
```

Validation:

```text
python scripts/validate_llama_cpp_router_trace.py \
  .codex_tmp/pc-router-trace-mixtral-jsonl/router-events.jsonl \
  --expected-layers 32 \
  --require-kind selected_experts \
  --require-kind selected_weights \
  --require-kind selected_weights_norm
```

Result:

- validator: passed
- events: 96
- layers: 32
- event counts: `selected_experts=32`, `selected_weights=32`, `selected_weights_norm=32`
- top-k: 2
- unique experts seen: 8
- note: Docker reported `unexpected EOF` after the JSONL was already complete
  and valid

## Nemotron Cascade

Model:

```text
hf.co/bartowski/nvidia_Nemotron-Cascade-2-30B-A3B-GGUF:Q4_K_M
```

Ollama blob:

```text
/root/.ollama/models/blobs/sha256-916a371ed63edd6602da950df2c1eed5ff74bfb9dac2ece64c7cabaafb3e2922
```

Validation:

```text
python scripts/validate_llama_cpp_router_trace.py \
  .codex_tmp/pc-router-trace-nemotron-cascade-jsonl/router-events.jsonl \
  --expected-layers 23 \
  --require-kind selected_experts \
  --require-kind selected_weights \
  --require-kind selected_weights_norm
```

Result:

- validator: passed
- exit code: 0
- events: 69
- routed layers: 23
- event counts: `selected_experts=23`, `selected_weights=23`, `selected_weights_norm=23`
- top-k: 6
- selected expert tensor shape: `[6,3,1,1]`
- unique experts seen: 125

## Qwen3 Coder

Model:

```text
hf.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q3_K_S
```

Ollama blob:

```text
/root/.ollama/models/blobs/sha256-17d51f5310e9a598e5ac914d30f401fb2d1bc3b6a06a846919099eac09364ae1
```

Validation:

```text
python scripts/validate_llama_cpp_router_trace.py \
  .codex_tmp/pc-router-trace-qwen3-coder-jsonl/router-events.jsonl \
  --expected-layers 48 \
  --require-kind selected_experts \
  --require-kind selected_weights \
  --require-kind selected_weights_norm
```

Result:

- validator: passed
- exit code: 0
- events: 144
- layers: 48
- event counts: `selected_experts=48`, `selected_weights=48`, `selected_weights_norm=48`
- top-k: 8
- selected expert tensor shapes: `[8,1,1,1]`, `[8,2,1,1]`
- unique experts seen: 128

## Boundary

These are semantic route traces from patched direct llama.cpp against PC Ollama
model blobs. They are not Python `register_forward_hook()` results and they do
not prove expert residency control, paging, preload, eviction, or cleanup.

The raw JSONL files live under `.codex_tmp/` and remain generated run artifacts.
