# PC llama.cpp Mixtral Hook Smoke - 2026-06-26

## Result

The PC Ollama `dolphin-mixtral:8x7b` model is directly readable by stock
llama.cpp from the Docker Ollama model volume, and llama.cpp's existing debug
eval callback can expose MoE route tensors.

This is a route-tensor smoke, not the structured JSONL trace contract.

## Model Blob

Ollama maps `dolphin-mixtral:8x7b` to:

```text
/root/.ollama/models/blobs/sha256-5041ba4278429fe475782b889471b5ff065a6cce5c3a539bd61d1e457f1961de
```

The Docker volume is:

```text
open-webui_ollama -> /root/.ollama
```

## Command Shape

The smoke used stock `ghcr.io/ggml-org/llama.cpp:full-cuda` with a read-only
Ollama volume mount and the existing `/app/llama-debug` binary:

```text
docker run --rm --entrypoint /app/llama-debug \
  -v open-webui_ollama:/root/.ollama:ro \
  ghcr.io/ggml-org/llama.cpp:full-cuda \
  -m /root/.ollama/models/blobs/sha256-5041ba4278429fe475782b889471b5ff065a6cce5c3a539bd61d1e457f1961de \
  -p Route_probe -c 64 -b 4 -ub 4 -ngl 0 --no-warmup \
  --tensor-filter ffn_moe_topk \
  --tensor-filter ffn_moe_weights_norm
```

## Evidence

- exit code: 0
- `ffn_moe_topk-*` layer tensors printed: 32
- `ffn_moe_weights_norm-*` layer tensors printed: 32
- first top-k tensor shape: `{2,4,1,1}`
- prompt token count: 4

## Boundary

This confirms the PC Ollama Mixtral blob is direct-llama.cpp hook-readable. It
does not yet produce `memory-moe-bridge-v1` JSONL because the PC stock
llama.cpp image has binaries and libraries, but no headers or CMake for building
the structured trace patch in-place.

The next PC step is to build or package the patched `llama-moe-router-trace`
binary for the PC container/runtime, then rerun this same blob through the JSONL
validator.
