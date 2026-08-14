# DeviceControlBench

An **executable** benchmark for on-device agentic device control. Given a request and a set of
device tools, an agent must take the correct action(s) — or correctly decline / ask — with the
right arguments, under a safety-confirmation model. Success is judged by **device state**, not
string matching.

Design & rationale: [`../docs/paper/benchmark-design.md`](../docs/paper/benchmark-design.md).
Metric & statistics definitions: [`../docs/paper/eval-protocol.md`](../docs/paper/eval-protocol.md).

## Layout
```
bench/
  tools/catalog.json   canonical device tools (grounded in Apple APIs) + sensitivity
  build_seed.py        reproducibly emits the seed task set -> tasks/*.jsonl
  tasks/*.jsonl        the dataset (6 categories)
  sandbox.py           executable in-memory device state + tools from the catalog
  normalize.py         time/duration/entity normalization (makes matching well-defined)
  checkers.py          executable success checkers (state / read / refusal / clarify / gate)
  runner.py            run one task through AgentCore against a fresh sandbox
  metrics.py           TSR/TSA/AEM/Slot-F1/FVR/TurnEff + safety (CGR/ORR/UER)
  stats.py             bootstrap CIs, paired bootstrap, McNemar
  backends_hf.py       HuggingFace Transformers backend (portable quality eval)
  harness.py           run a model over the benchmark -> results/<model>/<cfg>/run.json
  gold_replay.py       CI: every task's gold behaviour must pass its own checker
  summarize.py         aggregate results/ -> leaderboard.md / .csv
```

## Quick start (portable, no GPU)
```bash
make gold                                   # validate benchmark integrity
make eval MODEL=Qwen/Qwen2.5-1.5B-Instruct  # run a real model (HF backend, CPU/GPU)
make summarize                              # build the leaderboard
```
Results are written under `results/` with a full manifest (model revision, seed, dataset hash,
decoding params) so every number is reproducible.

## On Apple Silicon (efficiency numbers)
```bash
pip install mlx-lm
make eval-mlx MODEL=mlx-community/Qwen2.5-3B-Instruct-4bit
```
Quality is backend-invariant at temperature 0 (same weights/quant → same decisions); MLX adds
the latency / throughput / memory / energy measurements the paper reports for on-device use.

## Status
Ships the **seed** set (48 tasks, 8 per category, hand-authored, gold-validated). Scaling to
v1.0 (≥500 tasks, multi-annotator, held-out split) is milestone M6 in the research plan.
