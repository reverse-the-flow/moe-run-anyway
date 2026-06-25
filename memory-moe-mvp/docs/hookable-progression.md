# Hookable Progression

The intended progression is:

1. Passive external observation.
2. Internal hookable semantic tracing.
3. Fork, runtime actuator, or controller work only after hookable evidence.

This keeps the project from jumping from request-boundary telemetry directly to
runtime paging claims.

## Level 1: Passive External Observation

The passive level observes a running backend from the outside. It uses the
sidecar and stock runtime probes to capture request timing, response metadata,
endpoint observability, log growth, and coarse system/GPU state.

This level can answer:

- Is the backend reachable and stable?
- What does the endpoint expose?
- What timing and resource signals are reproducible?
- Which artifact fields can Model Plane hand off reliably?

This level cannot answer:

- Which semantic experts were routed?
- Which expert tensors were resident?
- Whether expert loading, eviction, pinning, or paging occurred.

## Level 2: Internal Hookable Semantic Tracing

The hookable level is the current active focus. It has two valid hook tracks:

- PyTorch or Transformers-style MoE runtimes where router, gate, or MoE modules
  can be observed through forward hooks.
- llama.cpp/GGUF MoE runtimes where the same routing decision must be observed
  inside the inference engine after top-k expert selection and before expert
  dispatch.

Stock GGUF/Ollama endpoints are not Python-forward-hookable, but that does not
make them out of scope. They are engine-hook candidates. The missing work is a
llama.cpp patch, fork, or callback surface that emits selected expert ids and
scores as artifacts.

This level should produce:

- routed layer ids
- routed expert ids
- routing weights, probabilities, or logits
- routing entropy
- prompt family metadata
- token or window summaries
- hook failure reports when a runtime does not expose enough structure

The current implementation surface is:

- `moe_forward_probe.py`
- `run_forward_probe_demo.py`
- `run_transformers_forward_probe.py`
- `scripts/plan_hook_trace_capture.py`
- `scripts/plan_hookable_moe_attempts.py`
- `ForwardHookMoEProbe`
- synthetic no-model hook smoke artifacts
- guarded local-only Transformers dry-run planning
- PC/GX10 hookability attempt matrix
- llama.cpp engine-hook contract note

The next real blockers are target/runtime specific:

- Python hook track: a runnable local Transformers MoE with compatible runtime
  dependencies, or a cloud run with enough memory.
- llama.cpp hook track: a source checkout and minimal patch that can emit router
  events for a direct GGUF run. This now exists for GX10 Mixtral and Qwen3.

The safe local hookable preflight is:

```sh
python3 scripts/plan_hook_trace_capture.py --model-path /path/to/local/transformers-moe --label local-hookable-moe --json
```

That planner emits three command classes: synthetic hook smoke, local
Transformers dry-run, and deferred approved local hook trace. The dry-run
refuses remote identifiers, missing local directories, missing
`torch`/`transformers`, and Hugging Face token environment variables. The first
Python-track semantic routing artifact still needs the approved command without
`--dry-run`, pointed at a local Transformers-style MoE directory whose router or
gate modules expose expert ids, weights, or logits through forward hooks.

The first llama.cpp-track semantic routing artifact uses direct llama.cpp, not
stock Ollama. The initial patch uses `LLAMA_MOE_ROUTER_TRACE_FILE` and writes
JSONL route tensor events for Mixtral and Qwen3 under the same
`memory-moe-bridge-v1` boundary.

## Level 3: Fork, Runtime Actuator, Or Controller

Fork/controller work is deferred until Level 2 produces traces. Without
semantic routing traces, controller policy work is mostly speculative. Without
evidence about hook points and routing data shape, a runtime fork risks solving
the wrong problem.

Fork/controller work should start only when:

- hookable traces validate under `memory-moe-bridge-v1`
- the project can replay routing traces through candidate policies
- dense or full-runtime fallback behavior is recorded for comparison
- the missing live capabilities are explicit: routing visibility, residency
  observation, residency control, fallback, cleanup, and artifact export

## Current Next Steps

1. Keep passive sidecar as Phase 1 evidence, not as semantic proof.
2. Run the synthetic hook smoke whenever hookable code changes.
3. Keep the PC/GX10 hookability matrix current.
4. For Python hooks, dry-run `run_transformers_forward_probe.py` against a
   compatible local Transformers MoE path.
5. Validate the llama.cpp route JSONL against the shared trace contract.
6. Package the llama.cpp patch/run path into a repeatable launch card.
7. Capture a Python-track semantic routing artifact when a compatible
   Transformers MoE runtime is available.
8. Only then revisit controller policies or live expert-loading actuators.
