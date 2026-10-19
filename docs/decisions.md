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

---

## ADR-0007 — Follow Apple's platform conventions for the app layer
**Status:** Accepted
**Context:** The shipping app targets iOS/iPadOS/macOS; we want it to feel native and pass
review.
**Decision:** Build to Apple's standards throughout:
- **Swift Package Manager** for `AgentCore`, `MLXRuntime`, and tool modules; a single
  **multiplatform SwiftUI app** target (no UIKit/AppKit unless a specific API requires it).
- **SwiftUI** for all UI, following the **Human Interface Guidelines** (native navigation,
  system materials, Dynamic Type, Dark Mode, accessibility labels).
- **App Intents** as the system-action bridge (not private APIs or URL hacks); the app also
  exposes its own intents so Siri/Shortcuts can drive it.
- **Swift API Design Guidelines** for naming; `async/await` and structured concurrency for
  the loop and tool I/O; `Codable` for tool schemas.
- **Privacy on Apple's terms:** a **Privacy Manifest** (`PrivacyInfo.xcprivacy`), purpose
  strings for every permission, least-privilege entitlements, and no data collection.
- Minimum deployment targets chosen for App Intents + MLX: **iOS/iPadOS 17, macOS 14**.
**Why:** Native conventions are the lowest-risk path to a good UX and App Store approval, and
they match the user's directive to decide by Apple standards.
**Trade-off:** Drops pre-iOS-17 devices; acceptable given MLX/App Intents needs.

---

## ADR-0008 — Default tool set: Apple framework + App Intents capabilities
**Status:** Accepted  ·  *(resolves design §10 Q2)*
**Context:** We must pick which real capabilities to build first.
**Decision:** Prioritize the capabilities Apple exposes cleanly and users expect, in this
order:
1. **In-app tools** (all platforms) — the Phase 2 baseline.
2. **EventKit** — Reminders & Calendar (create/query).
3. **Timers/alarms & opening apps / deep links** via App Intents.
4. **Messages & Mail compose** (user presses send — Apple never lets an app send silently).
5. **Shortcuts** — run user-authored shortcuts; expose our own App Intents.
6. **macOS Accessibility / Apple Events** — cross-app control (Mac only, Phase 4).
Contacts, HomeKit, Maps, Focus follow as fast-followers, each permission-gated.
**Why:** These are the sanctioned, high-value actions; nothing here fights the sandbox.
**Trade-off:** No "send a text without confirmation" or arbitrary iOS app control — both are
Apple prohibitions, not gaps in our design.

---

## ADR-0009 — Distribution: TestFlight → App Store; notarized for macOS
**Status:** Accepted  ·  *(resolves design §10 Q4)*
**Context:** Need a distribution path per platform.
**Decision:** **TestFlight** for iOS/iPadOS/macOS beta; **App Store** as the primary release
channel. Keep a **Developer ID–notarized** direct build for macOS as a fallback, since the
macOS Accessibility/Automation features are easier to ship outside the App Store sandbox if
review friction arises.
**Why:** Standard Apple pipeline; App Store maximizes reach, notarization keeps a viable Mac
path for the powerful (and review-sensitive) automation features.
**Trade-off:** App Store review may constrain the macOS automation surface; the notarized
build hedges that.

---

> **Design §10 Q3 (prompt-only vs. fine-tune)** is already decided by **ADR-0004**:
> prompt-only first, LoRA only if the eval suite plateaus below target.
