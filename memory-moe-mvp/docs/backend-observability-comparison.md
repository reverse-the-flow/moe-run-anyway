# Backend Observability Comparison

Date: 2026-06-20

This note compares the phase-2 Calliope MoE artifacts recorded in
`data/calliope_moe_phase2_evidence.json`. The comparison is about what each
backend can prove today, not about final expert paging.

## Summary

| Target | Backend | Ready for request probe | Changed metrics | Response shape | What it proves | What it cannot prove |
| --- | --- | --- | ---: | --- | --- | --- |
| Dolphin Mixtral 8x7B GGUF | llama.cpp | yes | 9 | normal answer, `stop` | Stock llama.cpp baseline works and emits useful token/timing metrics. | Semantic expert ids, residency, or routing policy effects. |
| Qwen3 30B A3B GGUF | llama.cpp | yes | 7 | reasoning-only, `length` at 220 tokens | Non-Mixtral GGUF MoE path works and the probe preserves reasoning channels. | Semantic expert ids or answer quality without a better prompt/token profile. |
| Nemotron 3 Super 120B GGUF | llama.cpp | no | 0 | model load failed | Current llama.cpp image reaches tensor loading and exposes a concrete compatibility error. | Request behavior, metrics, routing traces, or paging feasibility. |
| Nemotron 3 Nano Omni 30B A3B NVFP4 | vLLM | yes | 60 | normal content, `stop` at 512-token cap | vLLM is a viable OpenAI-compatible MoE baseline with richer metrics than stock llama.cpp. | Semantic expert ids or residency control from the stock OpenAI-compatible surface. |

## Interpretation

The llama.cpp path is still the right low-friction regression target. Mixtral
and Qwen3 both prove request handling, response capture, shared-contract shape,
and coarse runtime telemetry. The metric surface is compact and useful for
latency/token accounting, but it is not a router trace.

Qwen3 is important because it behaved differently from Mixtral: the first
successful request returned useful content in the reasoning channel with empty
message content. That justified preserving `reasoning_content` separately in
the runtime probe and sidecar summaries.

Nemotron Super is a compatibility blocker, not a runtime-observability success
or a memory-pressure result. The current llama.cpp image rejected
`blk.1.ffn_down_exps.weight` because the expected and actual tensor shapes did
not match. More prompts will not help until `nemotron_h_moe` GGUF tensor layout
support is inspected or the backend image is changed.

Nemotron Omni through vLLM is the strongest signal that MoE Run Anyway should
keep the OpenAI-compatible backend path. The vLLM metrics include request
success, prompt tokens, generation tokens, prefill/decode timing, time to first
token, and end-to-end latency. That is a richer baseline surface than stock
llama.cpp, while still falling short of semantic expert ids.

## Harness Direction

The next implementation work should split cleanly:

1. Keep llama.cpp baselines for Mixtral and Qwen3.
2. Keep vLLM baselines for Nemotron-H/HF-style MoEs.
3. Treat Nemotron Super GGUF as a backend compatibility task.
4. Move semantic expert-id work to a hookable runtime or backend patch.

Timing, token counts, and backend metrics are evidence that a request ran.
They are not evidence that we observed which experts routed, which experts
were resident, or whether a paging policy would have made the same decision.
