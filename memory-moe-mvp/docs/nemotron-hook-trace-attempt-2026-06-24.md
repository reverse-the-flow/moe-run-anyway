# Nemotron Hook Trace Attempt - 2026-06-24

## Target

- Model path: `/mnt/Calliope/models/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4-hf`
- Runtime image used for hook attempt: `vllm/vllm-openai:v0.20.0`
- Probe runner: `run_transformers_forward_probe.py`
- Prompt scope: `max_prompts=1`, `repeats=1`, `max_new_tokens=1`
- Trust boundary: local model directory only, `local_files_only=True`, no token env vars present.

## Why This Target

Gemma was rejected as a hook target because its config reports `enable_moe_block False` and no active expert metadata.

Nemotron is the correct local hook target:

- `llm_config.n_routed_experts: 128`
- `llm_config.n_shared_experts: 1`
- `llm_config.num_experts_per_tok: 6`
- local trusted-code files are present, including `modeling.py` and `modeling_nemotron_h.py`

## What Worked

The synthetic forward-hook smoke succeeded:

- run dir: `forward-probe-runs/20260624-140650-nemotron-preflight-synthetic-hook-smoke`
- `hook_count: 2`
- `router_event_count: 4`
- `token_count: 12`
- `unique_experts_seen: 5`
- sources: `expert_indices` and `router_logits`

The initial guarded Transformers dry-run succeeded inside the vLLM image, but it did not detect trusted-code import dependencies before the live run. That preflight gap has been fixed.

## Live Hook Attempt Result

The approved local hook trace did not reach model execution. It failed while importing local trusted model code:

```text
ModuleNotFoundError: No module named 'mamba_ssm'
ImportError: mamba-ssm is required by the Mamba model but cannot be imported
```

After adding the trusted-code dependency scanner, the dry-run correctly blocks before model load:

```text
trusted remote-code dependency not detected: causal_conv1d referenced by modeling_nemotron_h.py:69
trusted remote-code dependency not detected: mamba_ssm referenced by modeling_nemotron_h.py:57, modeling_nemotron_h.py:58, modeling_nemotron_h.py:64
```

## Environment Blocker

The existing vLLM container has `torch`, `transformers`, and `accelerate`, but not:

- `mamba_ssm`
- `causal_conv1d`

A bounded package-download feasibility check failed because the container could not resolve `pypi.org`, so the missing packages cannot currently be added from PyPI in this environment.

## Current State

This run produced hook pipeline evidence and a concrete dependency blocker. It did not produce a real Nemotron semantic router trace.

The next successful real-model hook trace needs one of:

- a hook runtime image that already includes `mamba-ssm` and `causal-conv1d`
- restored package/network access for building a derived hook image
- a smaller local Transformers MoE target without Mamba/custom-kernel dependencies

Until one of those exists, expert-paging work should stay behind the hook-trace gate.
