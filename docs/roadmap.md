# DeviceControlSlm — Roadmap

Phased plan. Each phase has a goal, the concrete work, and a **Definition of Done (DoD)** —
a checkable outcome. We do not start a phase before the previous DoD is met. Effort estimates
assume one focused developer; treat them as relative, not calendar promises.

---

## Phase 0 — Spike & feasibility (~1 week)
**Goal:** prove an SLM runs on the real target hardware at usable speed.
- Run the `mlx-swift-examples` LLM demo on a real iPhone/iPad and on a Mac.
- Load Qwen2.5-1.5B-Instruct (4-bit) on phone, Qwen2.5-3B (4-bit) on Mac.
- Measure tokens/sec, time-to-first-token, and **peak memory vs. jetsam budget**.
- In parallel, run the same model under `mlx-lm` (Python) on a dev machine.

**DoD:** documented tokens/sec + peak RAM per device tier; a go/no-go on model sizes; the
Python and on-device runners produce comparable output for the same prompt.

---

## Phase 1 — Agent core + eval in Python (~2 weeks)
**Goal:** a working tool-calling loop we can measure and harden fast.
- Implement the ReAct-style loop, tool protocol, JSON-schema registry, and prompt templates
  in `python/`.
- Build 4–6 **mock tools** (fake reminders, timer, messages, search).
- Stand up the **eval suite**: scripted tasks, tool-call validity + task-success metrics.
- Add **constrained JSON decoding** and the validate-and-repair retry.

**DoD:** ≥95% tool-call validity and a baseline task-success number on the eval suite; prompt
+ schema files factored out as shared data for reuse by the Swift port.

---

## Phase 2 — Swift core + on-device inference + minimal app (~3 weeks)
**Goal:** the validated agent, running on-device, with a chat UI.
- `AgentCore` SwiftPM package: port the loop, tool protocol, registry (shares Phase 1
  prompts/schemas).
- `MLXRuntime`: wrap `mlx-swift` inference (streaming, KV-cache, device-tiered model load).
- Multiplatform SwiftUI app: streaming chat, model download/management, a "stop" control.
- Wire in the **in-app tools** (adapter family A) so actions are real end-to-end.

**DoD:** on a real device, type a request → agent calls an in-app tool → observable result,
fully offline; parity with Python on the shared eval tasks that use in-app-equivalent tools.

---

## Phase 3 — OS actions via App Intents / Shortcuts (~2–3 weeks)
**Goal:** real system-level actions on iOS/iPadOS/macOS within Apple's APIs.
- `AppIntentsBridge` adapter (family B): reminders/calendar (EventKit), timers/alarms,
  message/mail compose, open app / deep link, run user Shortcuts.
- Lazy OS permission requests with purpose strings; graceful denial handling.
- The **safety confirmation sheet** for every `.confirm` tool.

**DoD:** on device, "remind me to X at 5pm" (and 2–3 more real tasks) create actual system
artifacts, each behind a confirmation gate; denied permissions degrade cleanly.

---

## Phase 4 — macOS Accessibility control (~2–3 weeks)
**Goal:** drive other apps on the Mac like a human would.
- `MacAccessibility` adapter (family C): `AXUIElement` inspect/click/type; Apple Events /
  AppleScript for scriptable apps; CGEvent fallback.
- Accessibility + Automation permission onboarding flow.
- All such tools `.confirm`+, macOS-target-only.

**DoD:** on a Mac, the agent completes a cross-app task (e.g. read a value from app A, enter
it into app B) via Accessibility, each action confirmed.

---

## Phase 5 — Reliability, memory, and (optional) fine-tune (~3 weeks)
**Goal:** make it dependable, not just demo-able.
- Expand the eval suite to cover the real tools from Phases 3–4.
- Long-term memory store (opt-in) + retrieval.
- **Optional LoRA/QLoRA fine-tune** on agentic/tool-calling data (`mlx-lm` LoRA), then
  quantize — only if prompt-only reliability plateaus below target (ADR-0004).
- Prompt-injection hardening: tool output treated as untrusted upstream of the safety gate.

**DoD:** target tool-call validity and task-success met across all real tool categories;
if fine-tuned, a measured improvement over the prompt-only baseline on held-out tasks.

---

## Phase 6 — Packaging & polish (~2 weeks)
**Goal:** shippable.
- Model download/update/management UX; storage and first-run flow.
- Battery/thermal-aware defaults; settings for model tier and step/time budgets.
- On-device-only diagnostics; crash/telemetry that never leaves the device.
- Distribution: TestFlight (iOS/iPad), notarized build or App Store (macOS).

**DoD:** a build installable on all three device classes via the chosen channel, passing the
full eval suite, with no network dependency for inference.

---

## Sequencing notes
- Phases 0–2 are the critical path to a working demo; 3/4 can be reordered based on whether
  you want to show iPhone or Mac first (open question #1 in `design.md`).
- The Python eval harness (Phase 1) is reused through every later phase — it is the
  regression net, not throwaway.
