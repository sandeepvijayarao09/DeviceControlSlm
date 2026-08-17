# How we test — methodology & standards

Testing runs at two levels: **software correctness** and **model evaluation**. Both follow
recognized standards; remaining gaps are listed honestly at the end.

## 1. Software correctness
| Check | What it guarantees | Standard |
| --- | --- | --- |
| `python/tests/` (16 unit tests) | agent loop, parser, metrics, normalization, dataset integrity | unit testing |
| **Gold replay** (`bench/gold_replay.py`) | every task passes *its own* checker under the reference solution — a task with an inconsistent gold can't enter the set | executable oracle / property test |
| **Swift tests** (`DeviceControlKit`) | DSL parsing, recipe selection, confirmation gate | unit testing |
| **CI** (`.github/workflows/ci.yml`) | all of the above + the leakage gate on every push/PR; also asserts generated Swift is in sync with the catalog | continuous integration |

## 2. Model evaluation
- **Executable (functional) evaluation.** Success = the resulting **device state**, checked by
  code (`bench/checkers.py`), never string-matching model prose. This is the SWE-bench/τ-bench
  style of grounding, adapted to device control.
- **Metrics** (`docs/paper/eval-protocol.md`): TSR (primary), tool-selection, argument
  exact-match, slot-F1, format validity, turn efficiency, and safety metrics — CGR, over-refusal,
  and **UER (unsafe-execution rate)**.
- **Uncertainty & significance.** Every TSR carries a **bootstrap 95% CI**; model-vs-model
  comparisons use a **paired bootstrap** and **McNemar's test** (`bench/stats.py`) — we never
  compare bare point estimates.
- **Train / test separation.** Two splits:
  - `bench/tasks` — in-distribution seed (shares phrasing style with training).
  - `bench/tasks_ood` — **held-out, independently authored** with novel verbs and indirect
    intents ("cut the wifi", "buzz me at 6:30", "I'm heading into a meeting"). This is the split
    that measures generalization, not memorization.
- **Leakage gate.** `tools/check_contamination.py` fails CI if *any* test prompt (either split,
  incl. paraphrases) appears verbatim in *any* training corpus. It already caught real leaks
  (22 seed prompts in an early dcslm corpus; 1 OOD prompt) — both fixed.
- **Determinism & reproducibility.** Greedy decoding (temperature 0), fixed seeds, and a per-run
  **manifest** recording model revision, dtype, seed, dataset hash, platform, and wall time
  (`bench/harness.py`). Any run reproduces to the same numbers.
- **Datasheet** (`benchmark-design.md` §6, Gebru et al.) documents motivation, composition,
  collection, and ethics.

## 3. Why the OOD split matters (the honest part)
In-distribution TSR flatters a from-scratch model that has only ever seen our phrasing. Reporting
**both** in-distribution and held-out OOD makes the generalization gap explicit rather than
hiding it — see `results.md`. A single in-distribution number would be the classic "great on the
data it was built around" trap; the OOD split is what turns this from a demo into a measurement.

## 4. Known gaps (roadmap to full rigor)
- **Human-authored test set.** The OOD split is authored by us; the gold standard is prompts
  written by independent people. Next.
- **Multi-seed variance.** Training/eval currently report a single run; add ≥3 seeds and report
  mean ± std for headline claims.
- **Coverage & model card.** Add Python coverage measurement and a Mitchell-et-al. model card for
  the shipped SLM.
- **Robustness suite.** Typos, code-switching, and adversarial phrasings as a dedicated stress
  test.
