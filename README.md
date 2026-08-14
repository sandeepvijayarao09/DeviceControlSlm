# DeviceControlSlm

An on-device **agentic** system built around a **Small Language Model (SLM)** that runs
locally on Apple devices — iPhone, iPad, and MacBook — using Apple's
[**MLX**](https://github.com/ml-explore/mlx) framework.

The goal is *not* to build a general-purpose chatbot. It is to build a compact model +
runtime whose one job is **agentic device control**: understand a user request, plan,
call tools, and take real actions on the device — entirely on-device, private, and offline.

> **Status: building toward a conference paper** (benchmark + system). Implemented and passing:
> the portable **agent core** ([`python/`](python/)), **DeviceControlBench** — an executable
> benchmark ([`bench/`](bench/)), and a **publication-grade eval harness** with real metrics,
> bootstrap CIs, and reproducible run manifests. Real model numbers land in [`results/`](results/).
> The Swift/MLX on-device app and the Apple-Silicon efficiency study are next.

## Try it now (no GPU, no model)
```bash
make gold                                    # validate benchmark integrity (48/48 tasks)
make test                                    # agent-core + benchmark unit tests
make eval MODEL=Qwen/Qwen2.5-1.5B-Instruct   # run a REAL model over the benchmark (CPU/GPU)
make summarize                               # aggregate results/ -> leaderboard
```
On an Apple Silicon Mac, `pip install mlx-lm` then `make eval-mlx MODEL=...` for the real
on-device model plus latency/memory/energy numbers.

## Documents

| Doc | What it covers |
| --- | --- |
| [`docs/paper/research-plan.md`](docs/paper/research-plan.md) | The paper: contributions, research questions, statistical protocol, model sweep, ablations, threats to validity. |
| [`docs/paper/benchmark-design.md`](docs/paper/benchmark-design.md) | DeviceControlBench spec + datasheet. |
| [`docs/paper/eval-protocol.md`](docs/paper/eval-protocol.md) | Precise metric & statistics definitions. |
| [`docs/design.md`](docs/design.md) | System architecture: layers, model choices, agent loop, tool system, per-platform control, safety, privacy. |
| [`docs/roadmap.md`](docs/roadmap.md) · [`docs/decisions.md`](docs/decisions.md) | Phased build plan and decision log (ADRs). |
| [`bench/`](bench/) · [`python/`](python/) | The benchmark + eval harness, and the agent core. |

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
