# Calliope MoE Architecture Pass

Date: 2026-06-20

This pass tested the available MoE-like local targets under `/mnt/Calliope/models`
as phase-2 evidence inputs. The goal was not to prove expert paging yet. The
goal was to separate backend compatibility, observability surface, and semantic
expert-routing visibility before choosing the next runtime path.

The machine-readable evidence packet is
`data/calliope_moe_phase2_evidence.json`; the backend-level comparison is
`docs/backend-observability-comparison.md`.

## Result Matrix

| Target | Format | Backend | Expert shape | Outcome | Evidence |
| --- | --- | --- | --- | --- | --- |
| Dolphin Mixtral 8x7B | GGUF | llama.cpp sidecar/runtime baseline | 8 experts, top 2 routed | Passed smoke. Straightforward stock-runtime baseline target. | `/tmp/moe-llama-cpp-smoke/20260619-190524-dolphin-mixtral-8x7b-llama-sidecar` |
| Qwen3 30B A3B | GGUF | llama.cpp sidecar/runtime baseline | 128 experts, top 8 routed | Passed smoke. Response surfaced as reasoning-only until the probe captured reasoning channels. | `/tmp/model-plane-moe-test-runs/20260619-203514-qwen3-30b-llama-sidecar-reasoning-capture` |
| Nemotron 3 Super 120B | GGUF | llama.cpp sidecar/runtime baseline | 512 experts, top 22 routed, 1 shared | Failed at load with a tensor-shape mismatch in the current llama.cpp image. This is a compatibility blocker, not the same as an out-of-memory result. | Model Plane run `run-nemotron-3-super-120b-llama-sidecar-20260620T014705Z-b9a847b1` |
| Nemotron 3 Nano Omni 30B A3B NVFP4 | HF checkpoint | vLLM OpenAI-compatible runtime | 128 routed experts, top 6 routed, 1 shared | Passed readiness and live baseline probes through vLLM. A 512-token-cap rerun finished with `stop`, 351 completion tokens, and 6.4s warm latency. Metrics are rich, but semantic expert ids are still not exposed by the stock endpoint. | `/tmp/model-plane-moe-test-runs/20260619-211310-nemotron-omni-30b-a3b-vllm-512` |

Dense or control candidates were also inspected. Gemma 4 31B, Qwen3.6 27B,
and Hermes 3 70B did not expose MoE expert metadata in the inspected local
metadata/config, so they are useful controls rather than phase-2 MoE targets.

## What The Architectures Told Us

Mixtral is the cleanest stock llama.cpp baseline. It is the right place to keep
checking harness shape, launch cards, and request/metrics capture because the
model loads and behaves predictably.

Qwen3 30B A3B is a good second llama.cpp baseline because it stresses a
different MoE family and response shape. It already showed why the artifact
schema must preserve reasoning channels separately from normal message content.

Nemotron 3 Super 120B is the current llama.cpp compatibility blocker. The load
failure points at `nemotron_h_moe` tensor layout support in the serving image,
not at the sidecar, launch card, or prompt harness. The next action is runtime
compatibility inspection or a newer/forked llama.cpp path, not more prompt
traffic.

Nemotron 3 Nano Omni 30B A3B is the strongest evidence that the harness should
not stay llama.cpp-only. vLLM exposed `/v1/models` readiness and `/metrics`,
loaded the NVFP4 MoE path, and produced request-level metrics including prefill,
decode, token, latency, and cache counters. It still does not expose semantic
expert ids through the OpenAI-compatible surface.

## Current Harness Implications

- Keep Mixtral as the simplest stock llama.cpp regression target.
- Keep Qwen3 as the non-Mixtral GGUF MoE baseline and reasoning-channel test.
- Treat Nemotron Super GGUF as a backend-compatibility task before any paging
  claim.
- Treat Nemotron Omni vLLM as a first-class OpenAI-compatible baseline backend.
- Use per-run response token caps for reasoning-heavy models so successful
  probes do not end as artificial `length` failures. The first live use of
  `--request-max-tokens 512` converted the Nemotron Omni probe from `length`
  to `stop`.
- Do not infer semantic routing from timing, token counts, or metrics deltas.
  Semantic expert ids still require a hookable runtime or a backend patch.

## Next Achievable Steps

1. Add a small prompt profile for reasoning-heavy models so the answer channel
   is easier to distinguish from internal reasoning.
2. Build the thin hookable PyTorch runner around `ForwardHookMoEProbe` for a
   local HF MoE target.
3. Inspect llama.cpp support for `nemotron_h_moe` GGUF tensor layout before
   trying more Nemotron Super launches.
4. Produce the backend observability comparison from the Mixtral, Qwen, and
   Nemotron Omni artifacts.
