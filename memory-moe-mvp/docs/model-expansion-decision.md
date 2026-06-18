# Model Expansion Decision

Date: 2026-06-13

## Decision

Do not add more model targets yet.

The current target matrix is sufficient while the project works through
passive observation and hookable semantic tracing. The registry already covers
the runtime surfaces this project can test without downloads, authentication,
Docker, a live server, or GPU-heavy work:

- stock `llama.cpp` or OpenAI-compatible observation
- OpenAI-compatible runtime readiness from Model Plane profiles
- passive sidecar proxy observation
- hookable PyTorch MoE semantic routing
- small local MoE checkpoints
- Mixtral-style matrix anchoring

Adding more named model entries now would mostly create aspirational inventory.
The missing evidence is not another brand or checkpoint name; it is the first
real artifact from a user-started backend and, separately, a local hookable
checkpoint run.

## Evidence Inspected

- Root `README.md`: confirms the project is measurement-first and defers live
  model work until a running backend or local hookable checkpoint exists.
- `UPLOAD_READINESS.md`: confirms no live `llama-server`, sidecar, runtime
  probe, hookable checkpoint, Docker path, package install, or model download
  has been run in the uploadable validation path.
- `memory-moe-mvp/README.md`: documents the implemented local surfaces and
  separates Tier 0 shape checks from Tier 1, Tier 2A, and Tier 2B live gates.
- `memory-moe-mvp/data/model_target_registry.json`: contains required
  target classes, each with a primary probe, observable signals, deferred
  requirements, and command shapes.
- `memory-moe-mvp/model_target_registry.py` and
  `memory-moe-mvp/tests/test_model_target_registry.py`: validate target shape,
  unique ids, required class coverage, and summary counts without dependencies.
- `memory-moe-mvp/docs/model-target-test-plan.md`: already defines the next
  live commands for stock runtime observation, passive sidecar observation, and
  hookable semantic routing.

## Local Cache Observations

Only local Hugging Face cache directory names were inspected. No model files
were loaded, no models were downloaded, no token was used, and no authentication
was attempted.

Observed names:

- `models--allenai--OLMoE-1B-7B-0924`
- `models--hf-internal-testing--tiny-random-MixtralForCausalLM`
- `models--nvidia--Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4`

These names do not require a new target class today:

- OLMoE-style candidates are already covered by `small_local_moe`.
- Tiny random Mixtral-style candidates are already covered by `mixtral_style`
  and `hookable_pytorch_moe` when a compatible local runtime exists.
- Large OpenAI-compatible candidates remain class-level runtime evidence unless
  a hookable runtime or backend patch exposes semantic expert ids.

The `openai_compatible_runtime` target class exists to test Model Plane
profiles that expose vLLM, Ollama, or generic OpenAI-compatible readiness. It is
not a request to add more named model inventory.

## What Would Trigger More Targets

Add a new registry target only when a no-download test would become more
actionable because of it. Concrete triggers:

- A live `llama-server` or sidecar baseline produces artifacts whose observable
  signals do not fit any current target class.
- A local hookable checkpoint exposes router or expert modules that require a
  different semantic extraction path from the existing forward-hook target.
- A small cached model can be tested immediately but is not representable as
  Mixtral-style, OLMoE-style, generic hookable PyTorch MoE, or generic
  OpenAI-compatible observation.
- A backend family with materially different observability is available locally
  and can be exercised without downloads, credentials, Docker, or GPU-heavy
  setup.
- A live baseline shows that model-family-specific prompt suites are
  needed before controller replay can compare routing stability.

## Next Recommended Model Order

1. Run the existing Tier 0 dependency-free checks and synthetic hook smoke.
2. Select one small hookable MoE target for semantic routing traces. Prefer a
   local or cloud target that exposes router or gate modules through Python.
3. If the local environment already has compatible Python dependencies, try the
   cached tiny random Mixtral-style checkpoint as a hook-shape smoke through a
   future thin Transformers runner.
4. Try the cached OLMoE-style checkpoint next for a small semantic MoE routing
   baseline, still only from a local path and only if dependencies already
   exist.
5. Run a user-started stock `llama-server` Mixtral-style or other local MoE GGUF
   through `llama_runtime_probe.py` and `llama_sidecar.py` when an approved
   upstream backend exists.
6. Defer larger candidates such as the observed Nemotron-style cache entry until
   the smaller opaque and hookable paths have produced contract-compatible
   artifacts.

## Registry Impact

No registry or test-plan update is required for this decision. The current
matrix covers the observed local cache names at the class level, and the next
project risk is live evidence collection rather than model-list expansion.
