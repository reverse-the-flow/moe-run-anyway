# Hookable MoE Attempts On PC And GX10 - 2026-06-25

## Result

No real model has produced semantic router traces yet.

The current state is:

- Synthetic hook pipeline: works.
- GX10 Nemotron HF MoE hook attempt: real target selected, blocked by missing hook-runtime dependencies.
- PC MoE models: available as Ollama/GGUF runtime targets. They are not Python-forward-hookable, but they are engine-hook candidates if we instrument the underlying inference engine.
- PC Hugging Face cache: contains local Transformers directories, but they are not MoE.

The machine-readable matrix is `memory-moe-mvp/data/hookable_moe_attempt_matrix.json`.
Validate it with:

```bash
python3 scripts/plan_hookable_moe_attempts.py --json
```

## PC

The PC has useful runtime MoE targets in Docker Ollama, including Mixtral,
Nemotron, Qwen3-Coder A3B, Llama 4 Scout, and DeepSeek V3.1 GGUF models. These
are not Python-forward-hookable. They do not expose Python modules or
`register_forward_hook()`. They should not be dismissed as impossible, though:
if llama.cpp/Ollama is routing the MoE, the route decision exists inside the
inference engine. The missing piece is an engine-level hook, patch, fork, or
telemetry callback that emits router choices as artifacts.

The PC Hugging Face cache was also checked:

- `sentence-transformers/all-MiniLM-L6-v2`: dense BERT embedding model.
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`: dense Qwen2 CausalLM.

Neither is a MoE hook target.

## GX10

GX10 has the one real local Transformers-style MoE target:

- `/mnt/Calliope/models/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4-hf`
- 128 routed experts
- 1 shared expert
- 6 experts per token

The hook attempt used the existing `vllm/vllm-openai:v0.20.0` image. That image
has `torch`, `transformers`, and `accelerate`, but it lacks the local Nemotron
trusted-code dependencies:

- `mamba_ssm`
- `causal_conv1d`

The dry-run now catches those dependencies before a live model load.

Other GX10 MoE models are GGUF/llama.cpp runtime targets:

- `dolphin-mixtral-8x7b.gguf`
- `qwen3-30b.gguf`
- `nemotron-3-super-120b.gguf`

They remain useful for runtime baselines and sidecar artifacts. For semantic
routes, they require a llama.cpp engine hook rather than a Python forward hook.

## Next Productive Action

Do not broaden into generic Mamba support. The Mamba dependency issue is
Nemotron-specific for this project.

The productive next move is one of:

- build or obtain a Nemotron hook runtime image with `mamba-ssm` and
  `causal-conv1d`
- stage one small, current Transformers MoE canary without custom Mamba
  dependencies

For GGUF models that already run in llama.cpp/Ollama, the parallel productive
move is a minimal llama.cpp engine-hook spike: find the MoE routing site, emit
layer id, token/window id, selected expert ids, and scores to JSONL, then compare
that artifact contract with the existing Python forward-hook contract.
