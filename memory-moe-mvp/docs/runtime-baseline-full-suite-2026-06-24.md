# Runtime Baseline Full Suite - 2026-06-24

This run used Model Plane launch cards to start each runtime, wait for the
runtime health gate, run `scripts/run_live_baseline.py`, stop the runtime, and
aggregate the resulting `manifest.json`, `events.jsonl`, and `summary.json`
artifacts with `scripts/aggregate_runtime_baselines.py`.

The prompt suite was `memory-moe-mvp/data/mixtral_probe_prompts.json` with all
18 prompts and 2 repeats per prompt. These artifacts are runtime/request
evidence only. They do not expose semantic expert IDs or per-token router
choices.

| Model | Backend | Requests | Failures | Mean ms | Median ms | P90 ms | Observability |
|---|---|---:|---:|---:|---:|---:|---|
| qwen3-30b.gguf | llama_cpp | 36 | 0 | 9943.648 | 9697.209 | 11036.919 | metrics, props, slots |
| dolphin-mixtral-8x7b.gguf | llama_cpp | 36 | 0 | 28015.628 | 26367.532 | 39980.224 | metrics, props, slots |
| gemma4_31b_nvfp4 | vllm_openai_compatible | 36 | 0 | 18306.803 | 16116.614 | 30577.223 | metrics |

Run artifact directories on codexlab:

- `/tmp/model-plane-moe-test-runs/20260624-074331-qwen3-30b-llama-sidecar-full18x2`
- `/tmp/model-plane-moe-test-runs/20260624-075032-dolphin-mixtral-8x7b-llama-sidecar-full18x2`
- `/tmp/model-plane-moe-test-runs/20260624-081345-gemma4-31b-nvfp4-vllm-full18x2`
- Aggregate JSON: `/tmp/model-plane-moe-test-runs/full-suite-aggregate.json`
- Aggregate Markdown: `/tmp/model-plane-moe-test-runs/full-suite-aggregate.md`

Immediate interpretation:

- The llama.cpp sidecar path is the best current surface for hookable runtime
  work because it consistently exposes `/metrics`, `/props`, and `/slots`.
- vLLM gives broad runtime metrics and prefix-cache counters, but not
  llama.cpp-style slot or props surfaces.
- Qwen3 30B was the fastest successful sidecar runtime in this pass, but this
  does not imply better expert routing. It only establishes runtime viability
  and baseline request behavior.
- The next engineering step is internal hook instrumentation for router or
  expert-choice evidence. Managed expert loading should remain blocked until
  hook-level traces exist.

Validation:

- `python3 -m unittest discover memory-moe-mvp/tests`
- Result on codexlab: 81 tests passed.
