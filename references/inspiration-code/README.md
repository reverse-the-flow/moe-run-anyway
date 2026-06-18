# Inspiration Code Links

This folder is for MoE runtime and expert-offload references that can inform MoE
Run Anyway design. Do not vendor third-party code into this repository. Local
source checkouts belong under `references/inspiration-code/checkouts/`, which is
ignored by Git. Keep durable project notes here as links, design observations,
or small original summaries. Use these repositories for architecture, workflow,
and data-structure inspiration; do not copy implementation code or project text
unless the license and attribution requirements are clear.

## Local Scratch Checkouts

The intended local layout is:

```text
references/inspiration-code/
  README.md
  checkouts/          # ignored; third-party repos may be cloned here
```

## Closest Implementation References

- [mixtral-offloading](https://github.com/dvmazur/mixtral-offloading)
  - Mixtral-focused expert offloading demo for consumer hardware.
  - Relevant idea: keep only active or recently used experts near GPU memory and
    use cache/offload policy instead of loading the whole MoE at once.
  - Useful for: baseline expectations, expert-cache ergonomics, and deciding
    what MoE Run Anyway should measure before a runtime actuator exists.

- [Fiddler](https://github.com/efeslab/fiddler)
  - Research prototype for local MoE inference under tight GPU memory.
  - Relevant idea: when expert weights are not resident on GPU, move activations
    to CPU and compute expert outputs there instead of copying large weights for
    every routed expert.
  - Useful for: CPU/GPU sidecar design, activation-transfer accounting, and
    alternate fallback paths when expert paging is too slow.

- [MoE-Infinity](https://github.com/EfficientMoE/MoE-Infinity)
  - Hugging Face friendly MoE serving library with expert activation tracing,
    prefetching, caching, and host-memory offload.
  - Relevant idea: a Python-facing offload runtime can expose an OpenAI-style
    serving surface while keeping expert storage and cache policy explicit.
  - Useful for: OpenAI-compatible harness tests, pull/run UX, and practical
    single-GPU offload ergonomics.

- [llama.cpp](https://github.com/ggml-org/llama.cpp)
  - Portable local inference runtime across CPU and many GPU backends.
  - Relevant idea: high-portability target for a future MoE actuator, especially
    if the project needs to work on generic GPU computers and not only vLLM
    servers.
  - Useful for: runtime probe contracts, metrics/log paths, and possible fork or
    controller-hook investigation.

- [vLLM](https://github.com/vllm-project/vllm)
  - High-throughput OpenAI-compatible serving runtime with broad MoE support.
  - Relevant idea: likely server-side target for OpenAI-compatible MoE harness
    tests and future expert-paging experiments inspired by FluxMoE/ReMoE.
  - Useful for: API compatibility, scheduler behavior, expert-parallel serving,
    and GPU-box deployments.

- [Flash-MoE](https://github.com/danveloper/flash-moe)
  - Apple Silicon/Metal prototype for running Qwen3.5-397B-A17B from SSD on a
    48 GB MacBook Pro by streaming only routed experts.
  - Relevant idea: pack experts into contiguous per-layer files, load the active
    experts with parallel positional reads, keep non-expert weights memory
    mapped, and use per-layer timing to separate routing, expert I/O, and GPU
    compute costs.
  - Useful for: managed expert loading design, measurement targets for an
    expert paging actuator, and negative results around custom caches,
    compression, prefetch hints, and speculative routing.
  - Caution: no repository-level license file was found in the checkout, so use
    it for architecture, workflow, and data-structure inspiration only unless
    licensing is clarified.

## Paper-First References To Watch
