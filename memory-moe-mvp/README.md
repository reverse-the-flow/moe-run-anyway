# Memory-Aware MoE MVP

This is a focused, testable MVP for a memory-aware Mixture-of-Experts runtime. It treats the problem as runtime orchestration first, not model redesign.

## What Is Being Optimized

The runtime faces a constrained optimization problem:

- Hard constraint: stay within one fixed memory budget.
- Primary objective: maximize a quality proxy.
- Secondary objective: minimize latency once quality stays acceptable.

The quality proxy in this MVP is intentionally simple and inspectable:

- expert quality: average hidden utility of the selected experts
- context quality: fraction of the prompt that fits into the current KV-cache budget
- combined quality: `expert_weight * expert_quality + context_weight * context_retention`

This makes the tradeoff measurable:

- keep more experts resident and future loads get cheaper
- keep more experts resident and context capacity shrinks
- evict more aggressively and long prompts fit better, but future loads get slower

## Minimal Architecture

Always resident:

- router metadata and policy logic
- shared/core model memory represented as `core_memory_mb`

Dynamically loaded:

- experts, each with `memory_mb` and `load_ms`

Placement in the stack:

- this MVP is a wrapper-style controller and simulator, not a modified transformer runtime
- it can later sit in front of a real runtime such as `llama.cpp` or another backend once expert load and unload hooks exist

## Passive Sidecar

There is now also a passive sidecar in [llama_sidecar.py](/X:/Experiments/memory-moe-mvp/llama_sidecar.py).

Its role is narrower than the simulator:

- proxy requests to a live `llama.cpp` or OpenAI-compatible backend
- log request boundary data such as endpoint, model, slot id, prompt size proxy, and streaming mode
- capture response summaries such as status, usage, timings, and finish reason when the backend exposes them
- capture before/after system memory snapshots and optional GPU snapshots
- optionally snapshot built-in `llama-server` endpoints such as `/metrics`, `/slots`, and `/props`
- write `manifest.json`, `events.jsonl`, and `summary.json` into a run folder

This is the non-invasive observability layer we need before deciding whether a backend fork or in-process hook is justified.

## Forward-Hook Probe

There is now a second, separate probe in [moe_forward_probe.py](/X:/Experiments/memory-moe-mvp/moe_forward_probe.py).

This one is for MoE stacks that still expose Python modules and forward hooks, such as a PyTorch or Hugging Face runtime. It is not for `llama.cpp`.

Its job is to log semantic routing traces that the passive sidecar cannot see:

- routed expert ids when the model exposes them
- selected expert weights or probabilities
- per-layer expert hit counts
- routing entropy summaries
- compact window summaries over hook events

It writes its own run artifacts:

- `manifest.json`
- `router_events.jsonl`
- `window_summaries.jsonl`
- `summary.json`

The intended use pattern is:

```python
from pathlib import Path
import moe_forward_probe

config = moe_forward_probe.ForwardHookProbeConfig(
    output_dir=Path("X:/Experiments/memory-moe-mvp/forward-probe-runs"),
    label="mixtral-transformers",
)
probe = moe_forward_probe.ForwardHookMoEProbe(config=config)
hook_count = probe.attach(model)

with probe.span({"prompt_id": "case-001", "family_id": "code_python"}):
    _ = model(**inputs)

probe.close()
```

Use this when the runtime still exposes router-like modules. If the backend is already opaque or fused beyond Python-level access, the next layer is a runtime-specific trace hook or an external profiler such as `Nsight Systems`, not more sidecar logic.

There is also a synthetic driver in [run_forward_probe_demo.py](/X:/Experiments/memory-moe-mvp/run_forward_probe_demo.py).

Use it when you want to exercise the second probe and generate real artifacts before a hookable MoE backend is wired up:

```powershell
py -3 X:\Experiments\memory-moe-mvp\run_forward_probe_demo.py
```

That demo does not measure a real model. It validates the forward-hook probe path itself by attaching to fake router modules and writing the same artifact shape a real PyTorch integration would use.

## Llama Runtime Probe

There is now also a separate `llama.cpp`-specific Probe 2 in [llama_runtime_probe.py](/X:/Experiments/memory-moe-mvp/llama_runtime_probe.py).

This is the appropriate second probe for stock `llama.cpp`. It does not try to use Python forward hooks because `llama.cpp` does not expose the model as Python modules. Instead it uses the observability surfaces that `llama-server` already provides:

- `/metrics`
- `/slots`
- `/props` when available
- server timing fields returned by completions
- optional per-request log growth from a configured `--log-file`

Its role is deeper than the passive sidecar but still non-invasive:

- submit requests directly to `llama-server`
- snapshot runtime state before and after each request
- correlate metric deltas with a prompt family
- capture log excerpts for the exact interval of that request

It writes its own run artifacts:

- `manifest.json`
- `events.jsonl`
- `summary.json`

Recommended `llama-server` flags for this probe:

```powershell
--metrics --slots --props --perf --log-file X:\path\to\llama-server.log --log-prefix --log-timestamps --verbosity 4
```

Example usage:

```powershell
py -3 X:\Experiments\memory-moe-mvp\llama_runtime_probe.py `
  --base-url http://127.0.0.1:18080 `
  --output-dir X:\Experiments\memory-moe-mvp\runtime-probe-runs `
  --label mixtral-runtime `
  --model dolphin-mixtral `
  --suite-path X:\Experiments\memory-moe-mvp\data\mixtral_probe_prompts.json `
  --max-prompts 4 `
  --repeats 1 `
  --log-file-path X:\path\to\llama-server.log
```

This probe still does not expose semantic expert ids. For that, the next step is a `libllama` harness or a small source patch that emits MoE routing events from inside the runtime.

## Probe Notes

A written note on the current probe surfaces, `llama.cpp` knobs, and when to use `server-cuda` versus `full-cuda` lives in [probe-observability-notes.md](/X:/Experiments/memory-moe-mvp/docs/probe-observability-notes.md).

A second design note on how to structure controllers from the sparse-attention insights lives in [controller-architecture.md](/X:/Experiments/memory-moe-mvp/docs/controller-architecture.md).

## Forward-Hook Container Stack

The pinned container stack for the Python / Transformers forward-hook path lives in:

- [forward-hook-probe.Dockerfile](/X:/Experiments/memory-moe-mvp/docker/forward-hook-probe.Dockerfile)
- [forward-hook-probe.requirements.txt](/X:/Experiments/memory-moe-mvp/docker/forward-hook-probe.requirements.txt)
- [forward-hook-probe.compose.yaml](/X:/Experiments/memory-moe-mvp/docker/forward-hook-probe.compose.yaml)

This stack is intentionally separate from the `llama.cpp` sidecar image. It is for the hookable PyTorch runtime path only.

## Mixtral PoC

The current real-backend proof target is `dolphin-mixtral:8x7b`.

Why this is the first target:

- true sparse MoE
- text-only, so the proof stays focused on runtime behavior
- clearly larger than a 16 GB VRAM budget
- simpler and more mature than newer multimodal MoE stacks

The passive sidecar is the first step in that path. It tells us what is already visible from a live Mixtral deployment before we decide whether to add runtime hooks or a fork.

## Expert Coverage Probe

The prompt corpus for the current `dolphin-mixtral:8x7b` probe lives in [mixtral_probe_prompts.json](/X:/Experiments/memory-moe-mvp/data/mixtral_probe_prompts.json).

This corpus is designed around one important constraint:

- we do not know the human-readable "domain" of each expert

So the probe does not assume there is a dedicated math expert, code expert, or legal expert. Instead it sweeps across routing-relevant input regimes:

- natural English prose
- Python code and tracebacks
- SQL and tabular inputs
- JSON, YAML, XML, and markdown structure
- math notation and LaTeX
- logs, regex, and shell commands
- multilingual text and non-Latin scripts
- hybrid prompts that mix several formats in one request

That is a better first probe because MoE specialization can emerge around token patterns, formatting, or language as much as topic.

Recommended protocol:

- run each probe prompt at least twice with `temperature=0`
- keep `max_tokens` modest so prompt-side routing differences are easier to compare
- start with the single-format prompt families before the hybrid families
- once expert IDs become visible through runtime hooks, compare both unique experts touched and stable co-activation pairs across repeats

You can load one prompt object from the corpus and post it through the sidecar with a small PowerShell loop:

```powershell
$suite = Get-Content X:\Experiments\memory-moe-mvp\data\mixtral_probe_prompts.json | ConvertFrom-Json
$probe = $suite.families[0].prompts[0]
$body = @{
  model = "dolphin-mixtral"
  messages = $probe.messages
  temperature = $suite.default_request.temperature
  top_p = $suite.default_request.top_p
  max_tokens = $suite.default_request.max_tokens
  stream = $suite.default_request.stream
} | ConvertTo-Json -Depth 8

Invoke-RestMethod `
  -Uri http://127.0.0.1:18080/v1/chat/completions `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

Before in-process hooks exist, success means we can at least:

- drive diverse prompt shapes through the same live Mixtral deployment
- log repeatable request timing and memory behavior for each probe family
- prepare a coverage-oriented workload for the moment expert IDs become observable

## Implementation Plan

The implementation is a lightweight simulation harness in [memory_moe.py](/X:/Experiments/memory-moe-mvp/memory_moe.py). It reuses patterns already present elsewhere in the workspace:

- inspectable run folders
- `manifest.json` for provenance
- `events.jsonl` for per-request decisions
- `summary.json` for aggregate comparisons

Core data structures:

- `ExpertSpec`: expert memory size and load cost
- `RequestCase`: prompt length plus router and quality score maps
- `RuntimeState`: current expert residency and LRU timestamps
- `PolicyProfile`: separate routing and residency policies

## Routing Policy

Implemented profiles:

- `baseline`: pure top-k by router score
- `reuse_bias`: top-k after adding a small resident bonus, a recency bonus, and a cold-load penalty
- `context_reserve`: same routing as `reuse_bias`, but paired with a more aggressive residency policy

This keeps routing and residency separate:

- routing decides which experts are desirable
- residency decides which experts stay in memory and which get evicted

## Residency Policy

Implemented policies:

- `lru`: only evict when a new selected expert does not fit
- `lru_context_reserve`: after selected experts are loaded, evict least-recently-used non-selected experts until the request's prompt fits or no more experts can be dropped

That second policy is the MVP version of reallocating freed expert memory to longer context.

## Evaluation Plan

Per profile, the simulator reports:

- mean and total latency
- mean load latency
- mean quality proxy
- mean quality ratio versus an oracle with full context
- mean context retention
- load count
- eviction count
- warm selection rate
- peak memory used

The aggregate comparison is intentionally Pareto-style instead of hiding everything inside one arbitrary scalar.

## Toy Experiment

The default scenario lives in [toy_workload.json](/X:/Experiments/memory-moe-mvp/data/toy_workload.json).

It uses:

- 6 experts with different memory sizes and load latencies
- 12 requests
- one fixed 1000 MB budget
- one fixed core memory footprint
- long prompts that punish over-caching experts
- revisits that reward reuse

The workload is designed to demonstrate all three behaviors:

- dynamic loading
- eviction
- a measurable context-capacity effect from expert residency

Concrete example from the default scenario:

- keeping `e0,e1,e2,e3` resident leaves about 4100 tokens of context capacity
- evicting back down to only `e0,e1` restores about 10100 tokens of context capacity

That is the exact systems tradeoff this MVP is meant to surface.

## Top 3 Risks

1. The quality proxy is synthetic.
   It is good for systems tradeoff testing, but not a substitute for real downstream evaluation.

2. Real expert load latency may be burstier than this model.
   Disk, PCIe, paging, and runtime synchronization costs can add non-linear effects.

3. Real routers may be less stable than the toy scores used here.
   If routing is noisy, a reuse-biased policy might amplify mistakes instead of only saving latency.

## How To Run

Print the plan:

```powershell
py -3 X:\Experiments\memory-moe-mvp\memory_moe.py plan
```

Run the simulation:

```powershell
py -3 X:\Experiments\memory-moe-mvp\memory_moe.py run
```

Run the passive sidecar in front of `llama-server`:

```powershell
py -3 X:\Experiments\memory-moe-mvp\llama_sidecar.py `
  --listen-port 8091 `
  --upstream-base-url http://127.0.0.1:8080 `
  --capture-gpu
```

Then point your client at `http://127.0.0.1:8091` instead of the upstream server directly.

If the upstream `llama-server` has observability endpoints enabled, you can also ask the sidecar to capture those before and after each proxied request:

```powershell
py -3 X:\Experiments\memory-moe-mvp\llama_sidecar.py `
  --listen-port 8091 `
  --upstream-base-url http://127.0.0.1:8080 `
  --capture-upstream-observability
```

Bundle the sidecar into a derived `llama.cpp` image:

```powershell
docker build `
  -f X:\Experiments\memory-moe-mvp\docker\llama-sidecar.Dockerfile `
  -t memory-moe-llama-sidecar:latest `
  X:\Experiments\memory-moe-mvp
```

For future `libllama` harness work, prefer the official `ghcr.io/ggml-org/llama.cpp:full-cuda` image instead of the server-only image. That image includes `libllama.so`, `llama-simple`, `llama-eval-callback`, and the rest of the broader tool surface.

Run the combined container for `dolphin-mixtral:8x7b` using your Ollama volume:

```powershell
docker run --rm `
  --name mixtral-sidecar `
  --runtime nvidia `
  --gpus all `
  -p 127.0.0.1:18080:8080 `
  -v open-webui_ollama:/root/.ollama:ro `
  -v X:\Experiments\memory-moe-mvp\sidecar-runs:/var/log/memory-moe-sidecar `
  -e SIDECAR_LABEL=dolphin-mixtral `
  memory-moe-llama-sidecar:latest `
  -m /root/.ollama/models/blobs/sha256-replace-with-dolphin-mixtral-blob `
  --ctx-size 8192 `
  --n-gpu-layers 24 `
  --batch-size 512 `
  --ubatch-size 128 `
  --cache-type-k q4_0 `
  --cache-type-v q4_0 `
  --flash-attn on
```

In this layout:

- `llama-server` listens only inside the container on `127.0.0.1:18080`
- the passive sidecar listens on container port `8080`
- the host connects to the sidecar on `127.0.0.1:18080`
- run artifacts are written to the mounted `sidecar-runs` directory

Run the tests:

```powershell
py -3 -m unittest discover -s X:\Experiments\memory-moe-mvp\tests
```

## What To Test Next

- Swap the synthetic quality proxy for a real task metric on a reduced MoE model or a stub backend.
- Add a profile that caps total resident expert memory directly instead of evicting only when pressured.
- Use the passive sidecar to identify what can be measured without a fork, then add minimal runtime hooks only where the sidecar cannot see enough.
