# PC Large GGUF MoE Preflights - 2026-06-26

## Result

The large PC Ollama MoE candidates were preflighted without attempting full
router traces in the current Docker Desktop VM. Docker memory is about 31 GiB,
while the oversized blobs are 51 GB, 63 GB, and 159 GB.

The stock llama.cpp metadata command was not safe enough for these sizes:

```text
/app/llama-gguf <blob> r n
```

It was killed with exit `137` on both large blobs after reading GGUF metadata
keys and much of the tensor table. A lightweight repo-local reader now handles
this preflight without loading tensor data:

```text
scripts/gguf_moe_preflight.py
```

## Llama 4 Scout

Model:

```text
llama4:17b-scout-16e-instruct-q4_K_M
```

Blob:

```text
/root/.ollama/models/blobs/sha256-9d507a36062c2845dd3bb3e93364e9abc1607118acd8650727a700f72fb126e5
```

Observed:

- blob size: 63 GB
- architecture: `llama4`
- tensor count: 1182
- block count: 48
- expert count: 16
- experts per token: 1
- routed MoE layers: 48
- MoE tensor evidence: `ffn_gate_inp`, `ffn_down_exps`, `ffn_gate_exps`, `ffn_up_exps`
- status: full patched router trace deferred to a higher-memory run window

## DeepSeek V3.1

Model:

```text
hf.co/unsloth/DeepSeek-V3.1-GGUF:TQ1_0
```

Blob:

```text
/root/.ollama/models/blobs/sha256-b66917dec6d7a9e9e923dbd083a0c98526baddf49e8f9a47116f6b86d283d363
```

Observed:

- blob size: 159 GB
- architecture: `deepseek2`
- tensor count: 1086
- block count: 61
- leading dense blocks: 3
- expert count: 256
- shared experts: 1
- experts per token: 8
- routed MoE layers: 58
- MoE tensor evidence: `exp_probs_b`, `ffn_gate_inp`, `ffn_down_exps`, `ffn_gate_exps`, `ffn_up_exps`
- status: full patched router trace deferred to a higher-memory host or run window

## Qwen3 Coder Next

Model:

```text
qwen3-coder-next:Q4_K_M
```

Blob:

```text
/root/.ollama/models/blobs/sha256-30e51a7cb1cf1333b9e298b90b4c7790fe2572d8736b002482a0ac96328a2ffb
```

Observed:

- blob size: 51 GB
- architecture: `qwen3next`
- tensor count: 843
- block count: 48
- expert count: 512
- experts per token: 10
- routed MoE layers: 48
- MoE tensor evidence: `ffn_gate_inp`, `ffn_down_exps`, `ffn_gate_exps`, `ffn_up_exps`
- status: full patched router trace deferred to a higher-memory run window

## Boundary

These are metadata and tensor-table preflights, not semantic router traces. They
prove local availability and MoE tensor structure. They do not prove selected
expert ids for a prompt, routing weights, runtime compatibility under the
patched runner, residency observation, or paging control.
