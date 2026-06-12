# Probe Observability Notes

This note records the observability surfaces and runtime knobs that are most relevant to the current probe suite.

## Probe Mapping

### Probe 1: Passive Sidecar

File: [llama_sidecar.py](../llama_sidecar.py)

Use when:

- the backend is already running behind `llama-server`
- we want request-boundary timings and coarse memory telemetry
- we do not want to modify model execution

What it can see:

- request path, prompt shape, model id, stream mode
- response status, usage, finish reason, timing fields returned by the server
- before/after system memory
- optional GPU snapshots
- optional upstream runtime endpoint snapshots when enabled

What it still cannot see:

- router logits
- selected experts
- per-layer dispatch
- expert residency

Recommended sidecar option:

- `--capture-upstream-observability`

That enables before/after snapshots of:

- `/metrics`
- `/slots`
- `/props`

### Probe 2: Llama Runtime Probe

File: [llama_runtime_probe.py](../llama_runtime_probe.py)

Use when:

- the target is stock `llama.cpp`
- we want deeper runtime correlation than a passive proxy provides
- we still want to avoid a custom `libllama` harness or source patch

What it can see:

- metric deltas from `/metrics`
- slot state before and after a request
- global properties from `/props`
- server timing fields from the completion response
- optional per-request growth of a configured `--log-file`

What it still cannot see:

- semantic expert ids
- per-layer routed top-k experts
- MoE routing mass

### Probe 3: Forward-Hook Probe

File: [moe_forward_probe.py](../moe_forward_probe.py)

Use when:

- the model still exists as Python modules
- the runtime exposes `register_forward_hook()`-style module hooks
- we want semantic routing traces

What it can see:

- explicit routed expert ids if the module exposes them
- selected expert weights or probabilities
- per-layer expert hit counts
- compact window summaries

This is not appropriate for stock `llama.cpp`.

### Next Level: libllama Harness

Use when:

- Probe 1 and Probe 2 show enough runtime structure to justify deeper work
- stock server endpoints are not enough
- we need in-process instrumentation

The likely `libllama` surfaces are:

- log callbacks via `llama_log_set(...)`
- perf helpers such as `llama_perf_context_print(...)`
- memory breakdown helpers such as `llama_memory_breakdown_print(...)`
- evaluation callback fields such as `cb_eval` in `llama_context_params`
- targeted graph dump or trace patches for MoE internals

That is where semantic MoE routing events for `llama.cpp` would likely be added.

## Relevant llama.cpp Knobs

### Observability and Logging

- `--metrics`
- `--slots`
- `--props`
- `--perf`
- `--log-file`
- `--verbose` / `--verbosity`
- `--log-prefix`
- `--log-timestamps`
- `--verbose-prompt`
- `--slot-save-path`

These are the first knobs to enable for runtime probing.

### Memory and Placement

- `--gpu-layers`
- `--split-mode`
- `--tensor-split`
- `--fit`
- `--fit-target`
- `--kv-offload`
- `--cache-type-k`
- `--cache-type-v`
- `--cache-prompt`
- `--cache-reuse`
- `--kv-unified`
- `--ctx-size`
- `--batch-size`
- `--ubatch-size`
- `--flash-attn`

These matter because they change the runtime regime that the probes are measuring.

### MoE-Specific

- `--cpu-moe`
- `--n-cpu-moe`
- `--cpu-moe-draft`
- `--n-cpu-moe-draft`

These do not expose expert ids, but they do change where MoE weights live and therefore matter for memory-aware probing.

## Recommended Stock llama-server Launch For Probing

```powershell
/app/llama-server `
  --metrics `
  --slots `
  --props `
  --perf `
  --log-file /var/log/llama-server.log `
  --log-prefix `
  --log-timestamps `
  --verbosity 4 `
  -m /models/model.gguf
```

## Docker Images To Use

Per the official `llama.cpp` Docker docs:

- `ghcr.io/ggml-org/llama.cpp:server-cuda` for server-only probe work
- `ghcr.io/ggml-org/llama.cpp:full-cuda` for broader `libllama` and CLI tool work

`server-cuda` is enough for the passive sidecar and runtime probe.

`full-cuda` is the better base when we want to build a `libllama` harness or inspect more than just `llama-server`.

## Forward-Hook Container Stack

Files:

- [forward-hook-probe.Dockerfile](../docker/forward-hook-probe.Dockerfile)
- [forward-hook-probe.requirements.txt](../docker/forward-hook-probe.requirements.txt)
- [forward-hook-probe.compose.yaml](../docker/forward-hook-probe.compose.yaml)

This stack is for the PyTorch / Transformers path, not `llama.cpp`.
