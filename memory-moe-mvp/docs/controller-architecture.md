# Controller Architecture

This note distills the main controller-design lessons from [ChatGPT-MoE Sparse Attention Insights.md](/X:/Experiments/AI data/ChatGPT-MoE Sparse Attention Insights.md) into a reusable architecture for our MoE probe work.

The key idea is not "use sparse attention for MoE." The useful transfer is a controller pattern:

- keep the outer model API unchanged
- intercept one internal decision boundary
- add a cheap selection stage before the expensive stage
- preserve a dense or baseline fallback path
- measure speed, quality, and routing behavior separately

## Core Pattern

A good controller sits between observation and expensive execution.

Its job is not to replace the model. Its job is to:

- observe a useful internal signal
- form a small candidate set or action proposal
- apply a bounded intervention
- record whether that intervention was actually selective and actually useful

In control terms:

- observe
- shortlist
- act
- audit

## Controller Components

### Observer

The observer reads the cheapest useful signal available before the expensive stage.

Examples:

- request-boundary hints from the sidecar
- runtime state from `/metrics`, `/slots`, `/props`
- recent expert history
- router logits or top-k expert ids when a hookable runtime exists

The observer does not decide. It only exposes a state description.

### Profiler

The profiler exists to avoid hard-coding one global policy too early.

Its job is to learn which layers, requests, or regimes behave differently.

Examples:

- layer-specific routing concentration
- short-window expert persistence
- prompt-family-specific latency patterns
- reuse or churn frequency

The profiler is especially important because sparse interventions can look "good" or "bad" simply due to backend effects or uneven layer behavior.

### Selector

The selector turns observations into a bounded candidate set.

This is the first real controller decision boundary.

Examples:

- shortlist experts to keep resident
- shortlist experts to preload
- shortlist experts allowed for a constrained-routing experiment
- choose whether to stay dense or enter a reduced mode

The selector should expose its policy as knobs, not bury it inside opaque logic.

Examples of good knobs:

- shortlist size
- reuse bonus
- cold-load penalty
- per-layer budget
- hysteresis threshold
- burst pool size

### Actuator

The actuator applies the controller decision with minimal outer-interface change.

Examples:

- keep selected experts warm
- evict low-value experts
- preload a small expert frontier
- force dense fallback for a comparison run
- constrain eligibility in a simulation or reduced runtime

The actuator should stay narrow. If it changes too many things at once, it becomes hard to tell whether the controller or the backend caused the result.

### Auditor

The auditor proves that the controller actually did what it claimed.

This is the most important protection against self-deception.

It should answer questions like:

- was the path actually sparse, or did fallback make it effectively dense?
- did the prompt meaningfully differentiate behavior?
- did the backend dominate the measurement?
- did selection help enough to justify its own overhead?

Without the auditor, a controller can quietly benchmark the baseline while pretending to test an intervention.

## Negative Design Rules

The sparse-attention insights suggest five failure modes that should become hard design gates.

### 1. Fake Sparsity

The controller claims to prune, but the actual path still behaves like the dense baseline.

Guardrail:

- log the effective candidate set size
- log whether fallback restored dense behavior
- compare against an explicit dense run

### 2. Low-Information Stimuli

The prompts do not meaningfully differentiate routing or runtime behavior.

Guardrail:

- use prompts that vary syntax, language, format, and task shape
- avoid tiny repeated-text toy cases as the main evidence

### 3. Backend Dominance

The engine, kernel path, memory layout, or dispatch overhead becomes the real experiment.

Guardrail:

- separate selection cost from execution cost
- log backend/runtime state around each request
- escalate to deeper instrumentation only when the current probe says the mystery lives there

### 4. Metric Collapse

One number gets mistaken for the whole story.

Guardrail:

- always carry at least one speed metric
- always carry at least one quality metric
- always carry at least one routing or selection metric

### 5. Instrumentation Tax Confusion

The probe layer is expensive, so the idea is incorrectly judged to be bad.

Guardrail:

- distinguish probe overhead from controller value
- treat the first probe as an observability rig, not as the final optimized implementation

## Recommended Metrics

Every controller experiment should report at least one metric from each class.

### Speed

- total latency
- prefill latency
- generation latency
- stall time
- bytes moved

### Quality

- downstream task metric
- consistency proxy
- output validity for structured tasks
- dense-baseline delta

### Routing or Selection Behavior

- shortlist size
- warm-hit rate
- eviction regret
- routing entropy
- expert reuse across windows
- fallback frequency

## Mapping To The Current Repo

### Already Present

- Observer:
  - [llama_sidecar.py](/X:/Experiments/memory-moe-mvp/llama_sidecar.py)
  - [llama_runtime_probe.py](/X:/Experiments/memory-moe-mvp/llama_runtime_probe.py)
  - [moe_forward_probe.py](/X:/Experiments/memory-moe-mvp/moe_forward_probe.py)
- Prompt differentiation:
  - [mixtral_probe_prompts.json](/X:/Experiments/memory-moe-mvp/data/mixtral_probe_prompts.json)
- Initial selector and actuator ideas in simulation:
  - [memory_moe.py](/X:/Experiments/memory-moe-mvp/memory_moe.py)

### Still Missing

- a controller that proves it is actually sparse under intervention
- a dense-fallback comparison harness for every controller run
- explicit separation of selection cost from dispatch/runtime cost in one shared report
- a real semantic-routing trace on a live MoE backend

## Practical Controller Shape For This Project

The next real controller should be small.

A good first version would look like this:

1. Observer:
   collect prompt family, recent runtime state, and if available recent expert history
2. Selector:
   score a small candidate expert set or resident set using simple heuristic terms
3. Actuator:
   choose one of:
   - keep current set
   - preload shortlist
   - evict low-value experts
   - force dense fallback
4. Auditor:
   record:
   - was the candidate set actually reduced?
   - what did fallback do?
   - what changed in latency?
   - what changed in output or quality proxy?

That keeps the control surface narrow enough to learn from.

## Bottom Line

The architecture lesson from the sparse-attention material is:

**A controller should be a small, explicit decision layer with visible knobs, a bounded intervention, a safe fallback, and an auditor that proves the intervention was both selective and worth its cost.**

That is the pattern we should keep reusing, regardless of whether the target is attention, MoE routing, expert residency, or cache policy.
