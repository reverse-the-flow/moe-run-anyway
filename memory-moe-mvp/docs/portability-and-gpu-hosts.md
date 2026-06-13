# Portability And GPU Hosts

Date: 2026-06-13

This repository is intended to be portable to an ordinary machine with Python,
git, and, for live model work, a GPU-capable local model backend that the user
starts explicitly. The upload/readiness path does not require a GPU.

## Portable Today

These paths are portable and do not require model weights, Docker, Hugging Face
credentials, or GPU access:

- project readiness from the repository root:

  ```bash
  python3 scripts/check_project.py
  ```

- host preflight from the repository root:

  ```bash
  python3 scripts/check_host.py
  ```

- registry validation, unit tests, Python compilation, and stale primary-doc
  path checks inside `scripts/check_project.py`
- synthetic replay and hook-shape work that uses local fixtures instead of a
  live model

Tier 0 is intentionally CPU-safe. It validates project shape, contracts, and
fixtures before any machine-specific backend is involved.

## GPU Host Preflight

Run this before attempting live model probes on a GPU computer:

```bash
python3 scripts/check_host.py
```

The preflight reports:

- OS, platform, Python executable and version, and project root
- `git --version`
- NVIDIA/CUDA tooling through `nvidia-smi`, when available
- AMD/ROCm tooling through `rocm-smi` or `rocminfo`, when available
- optional command presence for `llama-server`, `docker`, and
  `huggingface-cli`

It does not authenticate, download, load a model, start Docker, start a model
server, or run GPU-heavy work. It exits successfully by default even when no GPU
tooling is present, because Tier 0 is allowed on CPU-only hosts.

Use this stricter gate only when a live GPU machine is required:

```bash
python3 scripts/check_host.py --require-gpu
```

JSON output is available for automation:

```bash
python3 scripts/check_host.py --json
```

## NVIDIA/CUDA Hosts

On NVIDIA machines, the preflight expects `nvidia-smi` to be on `PATH` and to
return basic GPU, driver, and memory information. That is enough to say the host
has visible NVIDIA/CUDA tooling. It is not proof that a specific model server,
quantization, context size, or checkpoint will fit.

Live `llama.cpp` probing still requires the user to start a compatible
`llama-server` with a local model file and observability enabled. The project
does not bundle or download that model.

## AMD/ROCm Hosts

On AMD machines, the preflight looks for `rocm-smi` and `rocminfo`. Either tool
returning successfully is enough to say the host has visible AMD/ROCm tooling.
As with CUDA, this does not prove that a specific backend build, model format,
or checkpoint is compatible.

Live probes on ROCm machines require a user-provided local backend that exposes
the same HTTP or hookable runtime surfaces used by the project.

## Live-Machine Dependencies

The following remain intentionally outside the upload/readiness check:

- a user-started `llama-server` or OpenAI-compatible local model server
- local model weights or GGUF files already present on the machine
- a hookable PyTorch-style MoE checkpoint and compatible Python environment
- backend-specific CUDA or ROCm builds
- optional Docker workflows
- semantic expert ids from stock `llama.cpp`, unless a future runtime patch or
  harness exposes routing events

Command shapes for live probes are maintained in
[model-target-test-plan.md](model-target-test-plan.md).

## Hugging Face Tokens

Do not commit, paste, or write Hugging Face tokens into repository files,
scripts, prompts, logs, or docs.

If a separate local workflow needs Hugging Face access, keep credentials outside
the repository. Use an environment variable such as `HF_TOKEN` for one shell
session, or use an external login tool such as `huggingface-cli login` on the
host. This project preflight only checks whether `huggingface-cli` exists; it
does not inspect authentication state.

## Why Readiness Does Not Require GPU

The readiness check is meant to prove the uploadable project surface is
coherent everywhere: source files compile, unit tests pass, the target registry
is valid, and primary docs do not contain stale machine-specific paths. Requiring
a GPU would make that check less portable and would hide ordinary documentation
or contract regressions behind machine availability.

GPU work begins after Tier 0 passes and after the host preflight confirms the
target machine has suitable local tooling.
