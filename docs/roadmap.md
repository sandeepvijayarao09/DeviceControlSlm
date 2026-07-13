# DeviceControlSlm — Roadmap

Phased plan. Each phase has a goal, the concrete work, and a **Definition of Done (DoD)** —
a checkable outcome. We do not start a phase before the previous DoD is met. Effort estimates
assume one focused developer; treat them as relative, not calendar promises.

---

## Phase 0 — Spike & feasibility (~1 week)
**Goal:** prove an SLM runs on the real target hardware at usable speed.
- Run the `mlx-swift-examples` LLM demo **natively on the Mac** (the primary surface — MLX does
  not run in the iOS/iPadOS Simulator; see design §3.4). Load Qwen2.5-3B (4-bit).
- Run the same model under `mlx-lm` (Python) on the Mac for the Python dev/eval loop.
- Confirm the iPhone/iPad **Simulator** builds and runs the demo *UI* with a stub model backend.
- *(Optional, needs a physical iPhone/iPad)* load Qwen2.5-1.5B (4-bit) on device and measure
  tokens/sec, time-to-first-token, and **peak memory vs. jetsam budget**. Can be deferred to
  late Phase 2 if no device is available yet.

**DoD:** documented Mac tokens/sec + peak RAM; a go/no-go on model sizes; the Python and native
Mac runners produce comparable output for the same prompt; Simulator runs the UI on a stub model.

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

## Phase 2 — Swift core + native inference + minimal app (~3 weeks)
**Goal:** the validated agent, running natively on the Mac, with a chat UI.
- `AgentCore` SwiftPM package: port the loop, tool protocol, registry (shares Phase 1
  prompts/schemas).
- `MLXRuntime`: wrap `mlx-swift` inference (streaming, KV-cache, device-tiered model load),
  behind a `ModelBackend` protocol with a **stub backend** for the Simulator.
- Multiplatform SwiftUI app: streaming chat, model download/management, a "stop" control.
  Runs full-stack **on the Mac target**; runs UI-only on the iPhone/iPad Simulator via the stub.
- Wire in the **in-app tools** (adapter family A) so actions are real end-to-end.

**DoD:** on the **Mac**, type a request → agent calls an in-app tool → observable result, fully
offline with real MLX inference; the iPhone/iPad **Simulator** runs the same UI on the stub
backend; parity with Python on the shared eval tasks that use in-app-equivalent tools.
*(Optional: confirm real inference on a physical iPhone/iPad.)*

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
- **Mac-first** is the chosen path: because MLX can't run in the Simulator but runs natively on
  the Mac, the whole agent (inference + Accessibility control) is buildable and demoable on the
  developer's Mac with no extra hardware. Phase 4 (macOS Accessibility) can therefore land early.
- Phases 0–2 are the critical path to a working demo.
- The Python eval harness (Phase 1) is reused through every later phase — it is the
  regression net, not throwaway.
