# Research Plan — On-Device Agentic Device Control with Small Language Models

**Status:** living document · **Target:** full conference paper (Datasets & Benchmarks / Systems-for-ML)
**Center of gravity:** benchmark **and** system, equally weighted.

---

## 1. One-sentence thesis
Small (≤~4B) open language models, running fully on-device on Apple Silicon, can perform
*agentic device control* — interpret a request and emit the right **structured OS-command**
with the right arguments, taking safe multi-step actions **behind the scenes** (no screen
reading / no vision) — and we quantify how well, at what latency/memory/energy cost, and with
what safety guarantees, on a new executable benchmark.

**Scope (per project direction).** Control is *command-layer*, not screen-driven: the model
outputs structured system commands (like a command line without a terminal, which iOS/iPadOS
lack), executed by per-OS adapters — never by analysing screenshots or video. The benchmark
covers the everyday operations a person does by hand (connectivity, display, audio, power,
focus, alarms/timers, apps, media) plus user-app actions (reminders/calendar/messages). For
evaluating the *model*, the device is treated as **idealized/permissive** — every command is
executable in simulation; real per-OS permission gating (what Apple actually allows a
third-party app to do) is a separate, later engineering concern, not a property of the model.

## 2. Why this is publishable (novelty & positioning)
Existing tool-calling benchmarks are cloud/large-model oriented and not grounded in real,
safety-sensitive *device* actions:

| Benchmark | Focus | Gap we fill |
| --- | --- | --- |
| Berkeley Function-Calling Leaderboard (BFCL) | broad API function calling, mostly single-turn | no device grounding, no on-device cost, no safety/confirmation semantics |
| API-Bank | tool-use dialogue | large-model oriented; no executable device state; no on-device metrics |
| ToolBench / ToolLLM | massive real APIs | network APIs, not private on-device control; no efficiency frontier |
| τ-bench (tau-bench) | user-agent tool interaction in 2 domains | not device control; no on-device / small-model efficiency study |
| AppWorld, ToolACE | app/API control, data synthesis | not Apple on-device; no latency/energy on real silicon; no confirmation-gate metric |

**Our three contributions:**
- **C1 — DeviceControlBench:** an executable, safety-annotated benchmark for agentic device
  control (tools grounded in Apple frameworks: EventKit, App Intents, Messages/Mail compose,
  Shortcuts, macOS Accessibility), with state-based success checkers, paraphrase sets, and a
  datasheet. Categories include the cases prior work omits: **no-op/refusal**, **ambiguous →
  clarification**, and **safety-sensitive → confirmation**.
- **C2 — A safety-gated, portable agent runtime (AgentCore):** a compact ReAct-style loop with
  a **non-bypassable confirmation gate** for irreversible actions, dual-implemented in Python
  (reference/eval) and Swift (on-device), so the *same* agent is evaluated in the harness and
  shipped on the device.
- **C3 — An empirical quality×efficiency study on Apple Silicon:** a Pareto analysis across
  open SLMs and quantizations measuring task success **and** on-device latency / throughput /
  peak memory / energy, with statistical rigor and ablations.

## 3. Research questions
- **RQ1 (quality):** How accurately do ≤4B open SLMs select tools and arguments and complete
  multi-step device-control tasks? Where do they fail?
- **RQ2 (efficiency):** What is the on-device quality/latency/memory/energy Pareto frontier
  across model sizes and quantizations on Apple Silicon?
- **RQ3 (safety):** Can a runtime-level confirmation gate guarantee zero unsafe auto-execution
  while keeping over-refusal low, independent of model reliability?
- **RQ4 (method):** How much do constrained decoding, a repair retry, and few-shot exemplars
  each contribute (ablation)?

## 4. Metrics (defined precisely in `eval-protocol.md`)
**Quality:** Task Success Rate (TSR, *primary*, executable state check) · Tool-Selection
Accuracy (TSA) · Argument Exact-Match (AEM) & Slot-F1 · Turn Efficiency · Format Validity Rate
(FVR) · paraphrase-consistency.
**Safety:** Confirmation-Gate Recall (sensitive actions correctly gated) · Over-Refusal Rate ·
Unsafe-Execution Rate (must be 0 by construction).
**Efficiency (Apple Silicon):** time-to-first-token · decode tok/s · end-to-end task latency
(p50/p95) · peak RSS · energy per task (powermetrics / Instruments).

## 5. Statistical protocol (top standards)
- Primary decoding **temperature = 0** (deterministic); robustness runs at T>0 with **k≥5
  seeds**.
- **Bootstrap 95% CIs** over tasks (≥10k resamples) for every reported rate.
- Model-vs-model significance: **paired bootstrap** and **McNemar** on per-task success.
- Report **n**, seeds, and every hyperparameter. No cherry-picking; all runs land in `results/`.

## 6. Models (sweep)
Qwen2.5-{0.5B,1.5B,3B}-Instruct · Llama-3.2-{1B,3B}-Instruct · Gemma-2-2B-it ·
Phi-3.5-mini (3.8B) · SmolLM2-1.7B-Instruct. Quantization: **bf16/fp16 vs 4-bit**.
Reference ceilings (not on-device): Qwen2.5-7B-Instruct; a frontier API model as an *oracle
upper bound* only. Model identity pinned by **HF revision hash**.

## 7. On-device measurement protocol
- **Backends:** portable `llama.cpp`/GGUF for cross-platform *quality* (CI/Linux/Mac, identical
  weights → identical decisions at T=0); **MLX** on Apple Silicon for the *efficiency* numbers.
  Quality is backend-invariant at T=0 (same quant); we verify agreement on a sample.
- **Mac:** MLX + `powermetrics`/Instruments Energy Log; fixed thermal state (idle-cooled),
  plugged in, 3 warm-up runs, report median of ≥5.
- **iPhone/iPad (future work / add-on):** on-device via Swift `MLXRuntime`; same task set.

## 8. Building our SLM (fine-tuning, M5)
We do not pretrain; we **specialize** an open Apache-2.0 base via **LoRA** for structured device
control, targeting the baseline's measured failures (compositional collapse; acting instead of
asking). Pipeline in `finetune/`:
- **Data** (`gen_data.py`, no GPU): synthetic SFT trajectories in MLX chat format, using the
  *same* prompt/tool formatting as inference (no train/inference skew), with gold trajectories
  **rolled out through the sandbox** so multi-step examples teach act→observe→act→finish.
  **Contamination-controlled** against every benchmark prompt/paraphrase. ~2.4k examples,
  ~49% compositional / 27% single·read·comms / 14% refuse / 10% clarify; every gold action turn
  is verified to parse + validate.
- **Train/compare** (`train_and_eval.sh`, Apple Silicon): LoRA-fine-tune and fuse three
  candidate bases (Qwen2.5-0.5B, Qwen2.5-1.5B, SmolLM2-1.7B), reporting **BEFORE vs AFTER**
  benchmark metrics per base; the benchmark picks the winner (best TSR at acceptable
  size/latency), which becomes the shipped SLM.
- **Guarantee preserved:** the confirmation gate is a runtime property, so **UER stays 0**
  regardless of what the fine-tune changes.

## 9. Ablations
Constrained decoding on/off · repair-retry on/off · few-shot (0/2/4-shot) · prompt variants ·
tool-catalog size · **fine-tuned vs base** (headline) · data-mixture ablation (does upweighting
compositional trajectories drive the compositional gain?).

## 9. Threats to validity
Benchmark realism (mitigated: tools mirror real Apple APIs; state checkers, not string match) ·
annotation bias (multi-annotator + adjudication; datasheet) · quantization-vs-quality
confound (report both) · Mac-only device numbers (phone as explicit future work) · data
contamination (novel, held-out tasks; report release date).

## 10. Reproducibility & release
Pinned `requirements.lock`, model revision hashes, seeds, configs; results committed as JSON;
one-command `make eval`; plots regenerated from `results/`. Code under Apache-2.0; benchmark
under CC BY 4.0; datasheet + ethics statement (no real PII — synthetic contacts/notes).

## 11. Milestones → paper sections
| Milestone | Produces | Paper section |
| --- | --- | --- |
| M1 seed benchmark + executable checkers (this PR) | `bench/`, real first result | §Benchmark |
| M2 full model sweep (quality) | `results/quality/*` + tables | §Results (RQ1) |
| M3 Apple-Silicon efficiency sweep | `results/efficiency/*` | §Results (RQ2) |
| M4 safety-gate study | gate recall / over-refusal | §Results (RQ3) |
| M5 ablations | ablation tables | §Analysis (RQ4) |
| M6 scale dataset to 500+ w/ human validation | v1.0 dataset + datasheet | §Benchmark, appendix |
| M7 Swift on-device runtime + phone numbers | device results | §System, §Results |

## 11. Findings so far (answers to the RQs — see `results.md`)
- **RQ1 (can small models do device control?)** Yes, with limits. A fine-tuned 0.5B reaches
  51.6% in-distribution; a from-scratch ~4M model reaches 67.2% in-distribution.
- **RQ3 (safety):** the confirmation gate is a runtime invariant — **UER = 0%** across *every*
  model, independent of accuracy.
- **RQ4 (specialization):** LoRA nearly doubled the base 0.5B (29.7%→51.6%) and the effect
  reproduces on a second base (SmolLM2), so the gain is the method, not one base.
- **RQ5 (from scratch):** a purpose-built ~4M model tops the in-distribution leaderboard —
  narrow task ⇒ tiny model viable.
- **RQ6 (generalization, the key result):** on a held-out OOD split the ranking **flips** —
  the fine-tuned *pretrained* base (36.4%) beats the from-scratch model (32.7%); drop scales
  inversely with pretraining (−15 vs −34 vs −2). **Robust generalization comes from
  pretraining**, so the shipped product should specialize an open base; the from-scratch model
  is the tiny/ownable option pending more data or distillation.

**Threats to validity (current):** small OOD set (n=55, overlapping CIs — directions are the
robust claim); single training seed (variance bars pending); OOD authored by us (human-authored
set pending); quality track only (Apple-Silicon efficiency numbers pending).
