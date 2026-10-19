# DeviceControlSlm

## In 30 seconds
- **What it is:** A tiny AI model for iPhone, iPad and Mac that turns a request like "turn off Wi-Fi" into the actual device action, asking first before anything risky.
- **Why it matters:** Device control stays on the device, private, behind a safety check the model cannot skip.
- **Key result:** My from-scratch model (~4.1M parameters) completes 67.2% of benchmark tasks vs 51.6% for a fine-tuned Qwen2.5 model 120x its size; on unseen phrasing the larger model still wins (36.4% vs 32.7%). Unsafe actions: 0% for every model.
- **What I did:** Solo: model, tokenizer, training data, benchmark and Swift execution layer.
- **Built with:** Python, PyTorch, Hugging Face Transformers, LoRA, Swift/SwiftUI, App Intents, Core ML.
- **Try it:** `make test` (no GPU or Apple hardware needed).

An **on-device, agentic Small Language Model whose only job is device control** on iPhone, iPad,
and Mac. Not a chatbot: it maps a natural-language request to a structured OS command (or asks /
declines) and takes the action — locally, privately, under a non-bypassable confirmation gate.

The centerpiece is **DCSLM**, a transformer **built from scratch** (our own architecture,
tokenizer, and weights trained from random init on our own data — see
[`docs/paper/provenance.md`](docs/paper/provenance.md)). Pretrained open models (Qwen2.5,
SmolLM2), specialized with LoRA, are kept as honest baselines.

## Key results (DeviceControlBench, executable success, temperature 0)

| Model | Params | In-distribution | **Held-out OOD (n=55)** | UER |
| --- | --- | --- | --- | --- |
| DCSLM (from scratch) | ~4.1M | **67.2%** | 32.7% [20, 46] | 0% |
| Qwen2.5-0.5B fine-tuned | 494M | 51.6% | **36.4%** [24, 49] | 0% |
| Qwen2.5-0.5B base | 494M | 29.7% | 27.3% | 0% |

**The honest headline:** a tiny from-scratch model *dominates in-distribution* but generalizes
worst to novel phrasing — on held-out phrasing the fine-tuned **pretrained** base overtakes it.
Robust capability comes from pretraining. The **confirmation gate holds regardless** (unsafe-
execution rate = 0% for every model). Full analysis: [`docs/paper/results.md`](docs/paper/results.md).

## Repository

| Path | What |
| --- | --- |
| [`scratch/`](scratch/) | The from-scratch transformer (model, training, sampling). |
| [`dcslm/`](dcslm/) | Device-control corpus generator + the eval backend adapter. |
| [`finetune/`](finetune/) | LoRA specialization of open bases (portable HF + MLX). |
| [`bench/`](bench/) | DeviceControlBench: catalog, sandbox, checkers, harness, stats, `tasks/` (seed) + `tasks_ood/` (held-out). |
| [`python/agentcore/`](python/agentcore/) | Portable agent loop, parser, tool registry, safety gate. |
| [`apple/DeviceControlKit/`](apple/DeviceControlKit/) | Swift runtime + **App Intents / Shortcuts / Accessibility** execution layer. |
| [`apple/DeviceControlApp/`](apple/DeviceControlApp/) | Minimal SwiftUI app (iOS/iPadOS/macOS). |
| [`tools/`](tools/) | App Intents generator, Core ML export, contamination gate. |
| [`results/`](results/) | Run manifests + leaderboard. |

## Reproduce (no GPU, no Apple hardware)
```bash
make test           # agent-core + benchmark unit tests
make gold           # gold-replay both splits (every task passes its own checker)
make contamination  # train/test leakage gate (0 leaks)

# the from-scratch device SLM, end to end (CPU):
python -m dcslm.gen_corpus --n 40000          # build the corpus
python -m scratch.train --data dcslm/data/corpus.txt --out dcslm/out --iters 5000
python -m bench.harness --backend scratch --model dcslm/out/ckpt.pt                    # in-dist
python -m bench.harness --backend scratch --model dcslm/out/ckpt.pt --tasks-dir bench/tasks_ood
make summarize      # aggregate results/ -> leaderboard
```
On a Mac: `python tools/gen_appintents.py && swift test --package-path apple/DeviceControlKit`,
`python tools/export_coreml.py` for the Core ML model, then open `apple/DeviceControlApp` in Xcode.

## How it works
```
request → DCSLM → command DSL ("set_wifi|enabled=false ; set_brightness|level=30" / ASK / REFUSE)
        → agent loop → confirmation gate (safe/confirm/blocked)
        → per-OS recipe: Direct API → Shortcuts/App Intents → Accessibility → (unsupported)
```
The recipe is how iOS system toggles (no direct API) still work — via system Shortcuts actions.
One `catalog.json` drives the model's commands, the benchmark, and the App Intents (they can't drift).

## Testing (industry-standard)
Executable success (device state, not string match), bootstrap 95% CIs + McNemar, **held-out OOD
split**, a **contamination gate**, determinism + run manifests, and **CI on every push**
(unit tests, gold replay, leakage gate, Swift build/test). See
[`docs/paper/testing.md`](docs/paper/testing.md).

## Documents
| Doc | Covers |
| --- | --- |
| [`research-plan.md`](docs/paper/research-plan.md) | Contributions, RQs, statistics, ablations, threats. |
| [`dcslm-architecture.md`](docs/paper/dcslm-architecture.md) | The from-scratch model + system design. |
| [`benchmark-design.md`](docs/paper/benchmark-design.md) | DeviceControlBench spec + datasheet. |
| [`eval-protocol.md`](docs/paper/eval-protocol.md) · [`testing.md`](docs/paper/testing.md) | Metrics/stats; test methodology. |
| [`results.md`](docs/paper/results.md) | All findings (in-dist, OOD, fine-tune, cross-base). |
| [`model-card.md`](docs/paper/model-card.md) · [`provenance.md`](docs/paper/provenance.md) | Model card; what's from scratch. |

## Honest limitations
- OOD generalization gap (from-scratch model); small held-out set (n=55) → OOD numbers are
  directional. `read_query` = 0% by design. English-only. On-device execution controllers and
  Core ML export are validated by structure/CI but finalized on a Mac (they need Apple SDKs).
