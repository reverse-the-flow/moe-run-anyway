# Hookable MoE Attempts On PC And GX10 - 2026-06-25

## Result

Five real llama.cpp/GGUF runs have now produced semantic router traces through
a patched direct llama.cpp eval-callback hook.

The current state is:

- Synthetic hook pipeline: works.
- GX10 Mixtral GGUF engine hook: works.
- GX10 Qwen3 GGUF engine hook: works.
- GX10 Nemotron HF MoE hook attempt: real target selected, blocked by missing hook-runtime dependencies.
- PC Mixtral, Nemotron Cascade, and Qwen3 Coder Ollama blobs: work through the patched direct llama.cpp hook runner.
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
`register_forward_hook()`. The productive path is the llama.cpp engine-hook
track: read the Ollama blobs directly with a patched llama.cpp runner.

Validated PC direct llama.cpp hook traces:

- Mixtral: 96 events, 32 layers, top-k 2, 8 unique experts.
- Nemotron Cascade: 69 events, 23 routed layers, top-k 6, 125 unique experts.
- Qwen3 Coder: 144 events, 48 layers, top-k 8, 128 unique experts.

See `pc-llama-cpp-router-traces-2026-06-26.md`.

The remaining large PC Ollama candidates are present but should be scheduled
separately:

- Llama 4 Scout blob: 63 GB.
- DeepSeek V3.1 blob: 159 GB.

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

Mixtral and Qwen3 now have direct llama.cpp engine-hook traces:

- Mixtral run: `memory-moe-mvp/llama-cpp-hook-runs/20260625-mixtral-moe-router-trace-clean`
  - exit code 0
  - 96 JSONL events
  - 32 layers
  - event kinds: selected experts, raw selected weights, normalized selected weights
  - top-k shape: `[2,4,1,1]`
- Qwen3 run: `memory-moe-mvp/llama-cpp-hook-runs/20260625-qwen3-moe-router-trace-clean`
  - exit code 0
  - 144 JSONL events
  - 48 layers
  - event kinds: selected experts, raw selected weights, normalized selected weights
  - top-k shape: `[8,2,1,1]`

Nemotron Super GGUF remains a runtime compatibility target, not a completed
semantic trace target.

## Next Productive Action

Do not broaden into generic Mamba support. The Mamba dependency issue is
Nemotron-specific for this project.

The productive next move is one of:

- build or obtain a Nemotron hook runtime image with `mamba-ssm` and
  `causal-conv1d`
- stage one small, current Transformers MoE canary without custom Mamba
  dependencies

For GGUF models that already run in direct llama.cpp, the next move is to
package the patched Docker image and run command as a repeatable launch-card
path. PC Ollama itself is still unpatched, but its blobs are now proven readable
through direct patched llama.cpp.
