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

The hookable level is the current active focus. It targets PyTorch or
Transformers-style MoE runtimes where router, gate, or MoE modules can be
observed through forward hooks.

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
- `ForwardHookMoEProbe`
- synthetic no-model hook smoke artifacts
- guarded local-only Transformers dry-run planning

The next real blocker is target availability: a small local hookable MoE, a
user-provided checkpoint/runtime, or a cloud run with enough memory.

The safe local hookable preflight is:

```sh
cd memory-moe-mvp && python3 run_transformers_forward_probe.py --model-path /path/to/local/transformers-moe --output-dir forward-probe-runs --suite-path data/mixtral_probe_prompts.json --max-prompts 4 --repeats 1 --window-size-events 2 --dry-run
```

That command refuses remote identifiers, missing local directories, missing
`torch`/`transformers`, and Hugging Face token environment variables. The first
real semantic routing artifact still needs the same command without
`--dry-run`, pointed at a local Transformers-style MoE directory whose router or
gate modules expose expert ids, weights, or logits through forward hooks.

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
3. Select one small hookable MoE target.
4. Dry-run `run_transformers_forward_probe.py` against the local model path.
5. Capture one semantic routing artifact bundle.
6. Only then revisit controller policies or backend forks.
