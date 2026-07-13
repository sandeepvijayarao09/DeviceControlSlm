# DeviceControlSlm

An on-device **agentic** system built around a **Small Language Model (SLM)** that runs
locally on Apple devices — iPhone, iPad, and MacBook — using Apple's
[**MLX**](https://github.com/ml-explore/mlx) framework.

The goal is *not* to build a general-purpose chatbot. It is to build a compact model +
runtime whose one job is **agentic device control**: understand a user request, plan,
call tools, and take real actions on the device — entirely on-device, private, and offline.

> **Status: Phase 1 in progress.** The design is documented (below) and the **agent core +
> evaluation harness** are implemented and passing in [`python/`](python/) — runnable now with
> no GPU. The Swift/MLX on-device app (Phase 2+) is next. See the roadmap.

## Try it now (no GPU, no model)
```bash
cd python
python -m unittest discover -s tests -q   # agent loop / parser / safety gate tests
python -m eval.runner                      # eval suite: task success + tool-call validity
python demo.py --yes "remind me to call mom at 6pm"
```
On an Apple Silicon Mac, `pip install "mlx-lm>=0.19"` then add `--backend mlx` to run the
**real** small model. See [`python/README.md`](python/README.md).

## Documents

| Doc | What it covers |
| --- | --- |
| [`docs/design.md`](docs/design.md) | Full architecture: layers, model choices, agent loop, tool system, per-platform control, safety, privacy. |
| [`docs/roadmap.md`](docs/roadmap.md) | Phased build plan (Phase 0 → 6), milestones, and definition of done for each. |
| [`docs/decisions.md`](docs/decisions.md) | Decision log (ADRs) capturing the key choices and their trade-offs. |
| [`python/`](python/) | Runnable agent core + eval harness (Phase 1). |

## The short version

- **Model:** a 0.5B–3B open instruct model (e.g. Qwen2.5, Llama-3.2, SmolLM2), 4-bit
  quantized, sized to the device's RAM. Optional LoRA fine-tune later to harden tool-calling.
- **Runtime:** MLX — `mlx-lm` (Python) for fast development/eval, `mlx-swift` for on-device.
- **Agent:** a tight tool-calling loop (ReAct-style) with constrained JSON decoding, a small
  tool registry, short-term memory, and a hard safety/confirmation layer.
- **Device control, three scopes:**
  1. **In-app actions** — tools defined by the app, always allowed, no entitlements.
  2. **OS actions via App Intents / Shortcuts** — alarms, messages, calendar, opening apps
     (iOS, iPadOS, macOS — within Apple's APIs).
  3. **macOS Accessibility control** — drive other apps' UI on the Mac (click, type, read).
- **Shape:** one Swift package for the portable core, one multiplatform SwiftUI app,
  platform-specific tool adapters.

## Why this is realistic (and where the walls are)

Apple's sandbox means an iOS app cannot freely control the OS or other apps. What *is*
possible — and what this project leans on — is App Intents/Shortcuts and framework APIs
(EventKit, Contacts, HomeKit, MapKit, …) on iOS/iPadOS, plus the far more powerful
Accessibility and Apple Events APIs on macOS. The design is explicit about what each
platform can and cannot do, so we don't promise capabilities the OS won't grant.
