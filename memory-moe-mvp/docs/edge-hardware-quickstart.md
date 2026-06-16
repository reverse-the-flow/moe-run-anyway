# Edge Hardware Quickstart

MoE Run Anyway is still a measurement and routing harness, not a production
expert pager. Edge hardware makes that distinction more important: the practical
near-term path is routing among smaller models at the harness layer, then
connecting to true MoE expert paging only when a runtime actuator can prove
expert inventory, residency state, routing visibility, fallback, cleanup, and
artifact export.

Treat these as feasibility tiers, not guaranteed support. Model format,
quantization, runtime family, context length, KV-cache policy, batch size,
parallel slots, thermals, storage speed, and OS memory pressure can all move a
target between feasible, slow, unstable, and impossible.

The machine-readable companion is
[edge_hardware_quickstart.json](../data/edge_hardware_quickstart.json). Validate
it from the repository root with:

```bash
python3 scripts/plan_edge_hardware_quickstart.py
python3 scripts/plan_edge_hardware_quickstart.py --json
```

The planner is dependency-free and planning-only. It does not download models,
authenticate, run Docker, launch servers, inspect devices, or send prompt
traffic.

## Near-Term Shape

Small-model routing is easier to prove than live MoE expert paging.

Use the harness to route by:

- task family
- prompt family
- latency budget
- hardware budget
- context budget
- fallback policy

In this mode, small dense models act as routable experts or subagents. A route
can choose a small code model, a short-answer model, a summarizer, or a
fallback dense baseline without requiring runtime-level expert paging.

Evidence should stay explicit:

- router decision
- selected model class
- runtime family
- hardware tier
- context length and KV-cache settings
- artifact labels
- fallback use
- timing and readiness evidence

Do not claim semantic expert routing unless trace artifacts expose router
outputs, expert ids, and layer/window metadata. Endpoint timings, GPU memory
changes, or stock server metrics are runtime evidence, not semantic routing
evidence.

## Common Tiers

### CPU-Only / High RAM

Useful for compatibility baselines, slow smoke tests, and high-RAM quantized
experiments. Start with tiny or small dense models, then try larger dense or
sparse targets only after readiness artifacts are stable.

Minimum shape: 32GB system RAM for small dense smoke tests. For Mixtral-class or
DeepSeek-V2-Lite-class feasibility checks, expect 64GB+ and often 96GB to
128GB+ for a less fragile setup.

Candidate model classes:

- tiny smoke models
- small dense models as routable experts/subagents
- 7B to 14B dense quantized baselines
- small MoE or MoE-shaped smoke models
- DeepSeek-V2-Lite or Mixtral class only as high-memory feasibility targets

First test: plan baseline readiness, run a tiny/smoke route with short context,
then compare router decision and artifact labels.

### 8GB VRAM

Best treated as a small dense routing tier. It is not a natural starting point
for large MoE paging. Use one slot, short context, and conservative batch
settings first.

Candidate model classes:

- tiny smoke models
- small dense routable experts in the 0.5B to 7B class
- selected 7B or 8B dense quantized baselines

First test: compare two or three small dense route targets under the same prompt
family and latency budget.

### 12GB VRAM

Useful for small and medium dense route targets. Some sparse smoke tests may be
possible, but runtime support and quantization quality matter more than the
VRAM number alone.

Candidate model classes:

- tiny smoke models
- small dense routable experts
- 7B to 14B dense quantized baselines
- small MoE smoke or hookable toy MoE targets

First test: run a small-model routing matrix by prompt family, latency budget,
and artifact label. Defer larger sparse tests until baseline readiness is
boring.

### 16GB VRAM

This is a useful bridge tier: several small dense route targets, one larger
dense quantized target, or limited sparse feasibility smoke tests.

Candidate model classes:

- small dense routable experts
- medium dense models
- small MoE smoke models
- DeepSeek-V2-Lite-class targets only as feasibility candidates

First test: capture runtime baseline readiness, then test small dense expert
routing before trying a sparse target.

### 24GB VRAM

This tier can start to test larger sparse candidates, especially with short
context and quantization. It still should not be treated as proof of expert
paging.

Candidate model classes:

- small dense routable experts
- medium dense models
- small MoE smoke models
- Qwen3-30B-A3B class
- DeepSeek-V2-Lite class
- Mixtral class

First test: run a routed small-dense baseline first, then a short-context sparse
smoke load with artifact labels that separate runtime evidence from semantic
evidence.

### 48GB+ VRAM

Server/workstation memory makes larger sparse baselines plausible. It does not
create residency control or semantic routing evidence by itself.

Candidate model classes:

- small dense routable experts
- medium dense models
- hookable small MoE smoke models
- Qwen3-30B-A3B class
- DeepSeek-V2-Lite class
- Mixtral class

First test: capture baseline readiness, run a small routed suite, then compare
one larger sparse target with explicit trace-evidence labels.

### Apple Silicon Unified Memory

Unified memory removes a hard CPU/GPU split but not total memory pressure.
Thermals and other applications can still dominate sustained latency.

Minimum shape: 16GB unified memory for small dense smoke tests, 32GB+ for useful
local routing, and 64GB+ for larger sparse feasibility.

Candidate runtime families include `llama.cpp`, Ollama, MLX, Core ML, and
user-managed OpenAI-compatible endpoints. Keep runtime labels precise because
they expose different evidence.

First test: use small dense route targets first, then test larger sparse
candidates only with short context and clear memory labels.

### Jetson/ARM Edge Devices

Jetson and ARM edge targets are real edge devices, not just small servers.
Build compatibility, JetPack/CUDA/TensorRT versions, storage speed, power mode,
and thermal behavior are part of the test result.

Candidate model classes:

- tiny smoke models
- small dense routable experts
- small MoE smoke models only when the runtime exposes useful trace evidence

First test: run an edge smoke route with tiny or small dense models, recording
power mode, runtime family, context, and artifact labels.

### Android Phone/Emulator Targets

Phone emulators are useful for client compatibility, API wiring, UI packaging,
file access, and lightweight smoke tests. They are not substitutes for real
mobile GPU, NPU, memory bandwidth, battery, or thermal testing.

Use the emulator to prove that the client path can run and label artifacts.
Then repeat a tiny or small dense route on a real device before claiming mobile
feasibility.

Candidate runtime families:

- Android emulator client path
- `llama.cpp` Android builds
- MLC LLM
- ExecuTorch

Candidate model classes:

- tiny smoke models
- small dense routable experts

First test: emulator client/API smoke, then real-device tiny route with short
context and explicit thermal/device labels.

## RAM, Storage, Context, And KV Cache

Do not size a run from the model file alone.

Budget for:

- model weights
- quantization overhead and runtime buffers
- KV cache
- batch size
- parallel slots
- prompt and generated-token context
- OS memory and display memory
- model cache, conversion files, and artifact bundles

Storage should leave room for multiple model variants and failed-load artifacts.
Failed loads are useful evidence if they capture model, quantization, runtime,
hardware tier, context, and error labels.

Long context should be a separate test axis. First prove the route at tiny or
short context, then increase context after baseline readiness and artifact shape
are stable.

## Suggested Test Path

1. Capture baseline readiness from a saved manifest or planning artifact.
2. Run a tiny/smoke route only after explicit runtime approval.
3. Compare router decision, selected model class, fallback use, runtime family,
   timing, and artifact labels.
4. Keep small dense route targets as the near-term practical path.
5. Try Qwen3-30B-A3B, DeepSeek-V2-Lite, Mixtral, or other sparse targets only
   where memory headroom and runtime evidence make them plausible.
6. Do not claim semantic expert routing without trace evidence.

## Relationship To Expert Paging

Small-model routing and expert paging can share artifacts, labels, and
controller policy ideas. They are not the same capability.

Small-model routing chooses among separately runnable models or subagents. True
expert paging controls expert residency inside a sparse model runtime. The repo
should connect those paths only after runtime actuator evidence exists.
