# PC Ollama MoE Inventory Refresh - 2026-06-26

## Result

The current PC Ollama volume contains 13 MoE-class GGUF tags by metadata and
tensor-table scan. The scan used:

```text
scripts/gguf_moe_preflight.py
```

This reads GGUF headers, metadata, and tensor-info records without loading model
weights.

## Outcomes

| Model | Architecture | Size | MoE shape | Outcome |
| --- | --- | ---: | --- | --- |
| `dolphin-mixtral:8x7b` | `llama` | 26 GB | 32 layers, 8 experts, top-k 2 | semantic trace captured |
| `hf.co/bartowski/nvidia_Nemotron-Cascade-2-30B-A3B-GGUF:Q4_K_M` | `nemotron_h_moe` | 24 GB | 23 routed layers, 128 experts, 1 shared, top-k 6 | semantic trace captured |
| `hf.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q3_K_S` | `qwen3moe` | 13 GB | 48 layers, 128 experts, top-k 8 | semantic trace captured |
| `hf.co/unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_M` | `gemma4` | 17 GB | 30 layers, 128 experts, top-k 8 | semantic trace captured |
| `nemotron-3-nano:30b-a3b-q4_K_M` | `nemotron_h_moe` | 24 GB | 23 routed layers, 128 experts, 1 shared, top-k 6 | semantic trace captured |
| `qwen3-coder:30b` | `qwen3moe` | 19 GB | 48 layers, 128 experts, top-k 8 | semantic trace captured |
| `llama4:17b-scout-16e-instruct-q4_K_M` | `llama4` | 67 GB | 48 layers, 16 experts, top-k 1 | blocked: host memory preflight |
| `qwen3-coder-next:Q4_K_M` | `qwen3next` | 52 GB | 48 layers, 512 experts, top-k 10 | blocked: host memory preflight |
| `hf.co/unsloth/DeepSeek-V3.1-GGUF:TQ1_0` | `deepseek2` | 170 GB | 58 routed layers, 256 experts, 1 shared, top-k 8 | blocked: host memory preflight |
| `gpt-oss:20b` | `gptoss` | 14 GB | 24 layers, 32 experts, top-k 4 | blocked: llama.cpp tensor layout |
| `gemma4:26b` | `gemma4` | 18 GB | 30 layers, 128 experts, top-k 8 | blocked: llama.cpp tensor count mismatch |
| `qwen3.6:35b` | `qwen35moe` | 24 GB | 40 layers, 256 experts, top-k 8 | blocked: llama.cpp rope metadata mismatch |
| `glm-4.7-flash:Q8_0` | `glm4moelite` | 32 GB | 46 routed layers, 64 experts, 1 shared, top-k 4 | blocked: unsupported llama.cpp architecture |

The machine-readable source of truth is:

```text
memory-moe-mvp/data/hookable_moe_attempt_matrix.json
```

## New Semantic Traces

The 2026-06-26 refresh added three successful PC semantic traces:

- `hf.co/unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_M`: 90 events, 30 layers, top-k 8, 128 unique experts.
- `nemotron-3-nano:30b-a3b-q4_K_M`: 69 events, 23 routed layers, top-k 6, 124 unique experts.
- `qwen3-coder:30b`: 144 events, 48 layers, top-k 8, 127 unique experts.

## Boundary

This inventory is scoped to the PC Docker Ollama volume at scan time. Dense
models and GGUF files without expert metadata or MoE tensors are not listed
here. Host-memory blockers are not semantic trace failures; they mean the blob
is present and MoE-structured but should be traced on a larger memory budget.
Runtime compatibility blockers are direct patched llama.cpp load failures before
router events were emitted.
