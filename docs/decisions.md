# Decision Log (ADRs)

Short architecture decision records. Each captures the choice, why, and what we gave up.
Status: **Proposed** until the design is approved.

---

## ADR-0001 — Adapt an existing SLM; do not pretrain from scratch
**Status:** Proposed
**Context:** We need a small model good at agentic tool-calling on-device.
**Decision:** Start from an existing open instruct model (Qwen2.5 / Llama-3.2 class) and, if
needed, LoRA fine-tune it. Do not pretrain a new model.
**Why:** Pretraining a competitive model costs months of compute and large curated corpora,
and would still trail existing 1–3B instruct models on instruction following. The value we
add is the *agent runtime + tool reliability + device control*, not the base weights.
**Trade-off:** We inherit the base model's licence and biases; mitigated by keeping the model
behind an abstraction so it can be swapped.

---

## ADR-0002 — MLX as the on-device runtime
**Status:** Proposed
**Context:** User targets iPhone/iPad/MacBook (Apple Silicon).
**Decision:** Use MLX — `mlx-lm` (Python) for development/eval, `mlx-swift` for on-device.
**Why:** Native to Apple Silicon with strong performance and unified memory, first-class
quantization, an active LLM ecosystem, and a clean Python→Swift path that shares weights and
behavior. Best fit for an Apple-only target.
**Alternatives rejected:** *Core ML* — broader device reach but a heavier conversion pipeline
and less flexible for a fast-moving agent loop; kept as a fallback if we need older devices.
*llama.cpp/GGUF* — great cross-platform reach but less Apple-native and off-target here.

---

## ADR-0003 — Portable Swift core, platform-specific tool adapters only
**Status:** Proposed
**Context:** Three platforms with very different control capabilities.
**Decision:** Keep `AgentCore` (loop, tool protocol, registry, memory) platform-agnostic;
isolate all `#if os(...)` code in tool adapter targets and the app layer.
**Why:** Maximizes shared, testable logic; makes the sharp platform differences (esp.
macOS-only Accessibility) explicit and contained.
**Trade-off:** A little extra protocol/abstraction ceremony up front.

---

## ADR-0004 — Prompt-only first; LoRA fine-tune only if reliability plateaus
**Status:** Proposed
**Context:** Small models emit invalid tool calls more often than large ones.
**Decision:** Reach tool-call reliability first with **constrained decoding + schema
validation + few-shot**. Only invest in a LoRA/QLoRA fine-tune (Phase 5) if the prompt-only
baseline stays below target on the eval suite.
**Why:** Constrained decoding structurally prevents malformed JSON and needs no training data;
fine-tuning adds a data-collection and training burden we should only pay if measured need
justifies it.
**Trade-off:** Prompt-only may cap task-success for complex tools; the eval suite tells us
when to escalate.

---

## ADR-0005 — Hard safety gate between the model and any real action
**Status:** Proposed
**Context:** The agent takes real, sometimes irreversible, device actions; tool output and app
content are untrusted (prompt-injection risk).
**Decision:** Classify every tool `.safe` / `.confirm` / `.blocked`. `.confirm`+ actions
require an explicit user confirmation sheet showing tool + arguments before execution. The
model can *request* but never *unilaterally perform* a sensitive action.
**Why:** On-device agency plus a small (more steerable/injectable) model demands a
non-bypassable human checkpoint for anything with side effects.
**Trade-off:** More taps for the user on sensitive flows; acceptable for safety, and `.safe`
tools stay frictionless.

---

## ADR-0006 — Develop and evaluate in Python, ship in Swift, share the data
**Status:** Proposed
**Context:** Swift iteration on-device is slow; we need a tight measure/improve loop.
**Decision:** Build the agent loop and eval suite in Python (`mlx-lm`) first, then port to
Swift, with prompt templates and tool JSON schemas checked in as **shared data files** both
sides load.
**Why:** 10× faster iteration on the part that's hardest to get right (tool reliability), and
the shared schemas/prompts mean Python eval gains transfer directly to the shipped app.
**Trade-off:** The loop logic exists in two languages; kept in sync by sharing the data and
the eval suite as the contract.
