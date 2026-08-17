# Results (living) — DeviceControlBench v0.2 seed

First real runs on the reoriented OS-level benchmark. **Portable quality track** (HF
Transformers, CPU, greedy/temperature 0, deterministic). On-device efficiency numbers (MLX on
Apple Silicon) are pending. These are seed-set numbers (N=64); v1.0 (≥500) is milestone M6.

Dataset hash: `cf1ed9dcd23ebc0f` · reproduce with `make eval MODEL=... && make summarize`.

## Fine-tuning result (RQ4) — the headline
A **250-example LoRA** (Qwen2.5-0.5B, rank 16, trained on CPU here in ~45 min) nearly **doubles
task success** and, most importantly, **cracks the two categories the base scored 0% on**:

| Category | base 0.5B | **fine-tuned 0.5B** | Δ |
| --- | --- | --- | --- |
| **TSR (overall)** | 29.7% | **51.6%** | **+21.9** |
| compositional | 0% | **60%** | **+60** |
| ambiguous_clarify | 0% | **30%** | **+30** |
| no_op_refusal | 40% | 80% | +40 |
| single_tool | 86% | 93% | +7 |
| safety_sensitive | 30% | 30% | 0 |
| read_query | 0% | 0% | 0 |

Also: TSA 34→59%, AEM 40→64%, FVR 80→95%, over-refusal 55→18%, and **UER stays 0%** (the
runtime guarantee is untouched by fine-tuning). **The fine-tuned 0.5B (51.6%) beats the base
1.5B (45.3%)** — a specialized model 3× smaller than the base it surpasses. This is the core
"build our SLM" result: the base's failures are a *training* gap, not a capacity ceiling, and
even a tiny targeted LoRA closes most of it. `read_query` is the remaining weak spot (the model
calls the right query tool but doesn't reliably surface the retrieved value) — next data-mix
target. Reproduce: `make ft-data && python -m finetune.train_hf ... && make summarize`.

## From-scratch device-only SLM (RQ5) — narrow beats big
A transformer **built from scratch** (our code, random init) and trained **only** on
device-control DSL — **~4.1M params, ~120× smaller than the 0.5B** — tops the leaderboard:

| Model | Params | TSR (95% CI) | UER | speed |
| --- | --- | --- | --- | --- |
| **DCSLM (from scratch, device-only)** | **~4.1M** | **75.0% [64%, 86%]** | 0% | ~0.1s/task |
| Qwen2.5-0.5B — fine-tuned | 494M | 51.6% [39%, 64%] | 0% | ~12s/task |
| Qwen2.5-1.5B — base | 1.5B | 45.3% [33%, 58%] | 0% | — |

Per category: single_tool 100%, clarify 100%, refusal 100%, compositional 70%, safety 70%,
read_query 0%. FVR 95%, CGR 70%.

**Interpretation — and the honest caveat.** This validates the core thesis: when the task is
*only* device control, a tiny purpose-built model beats general models 100–350× its size, at a
fraction of the latency/energy — ideal on-device. **But** this is **in-distribution**
generalization: the training corpus and the benchmark share a phrasing *style* (both authored
around the same command set). The corpus is decontaminated of the benchmark's exact prompts +
paraphrases (verified 0 leaked), so this is not test-set memorization — but it does **not** yet
measure robustness to genuinely novel user phrasings. The needed next rigor step is a held-out
"hard paraphrase" test set written by a different process; a from-scratch model (no world
knowledge) will drop more there than a pretrained base would. `read_query` stays 0% by design
(the model picks the query tool but can't know the value; the runtime must fill it).

## Held-out OOD test (RQ6) — the result that reframes everything
Evaluating on `bench/tasks_ood` (independently authored, novel phrasing, verified 0 leakage):

| Model | Params | In-distribution TSR | **Held-out OOD TSR** | Δ |
| --- | --- | --- | --- | --- |
| DCSLM (from scratch) | ~4.1M | 75.0% | **35.5%** [19, 52] | **−40** |
| Qwen-0.5B fine-tuned | 494M | 51.6% | **32.3%** [16, 48] | −19 |
| Qwen-0.5B **base** | 494M | 29.7% | **32.3%** | **+3** |

**Reading this honestly:**
- The **base model is phrasing-invariant** (29.7% → 32.3%, within noise): its pretrained language
  understanding transfers to wording it never saw.
- The **specialized models were substantially memorizing phrasing.** On held-out phrasing the
  from-scratch model loses 40 points and its large in-distribution lead **collapses into a
  statistical tie** with the base (all three ≈ 32–36%, CIs overlap heavily at n=31).
- So the earlier "narrow-from-scratch beats big" headline is an **in-distribution** artifact.
  The robust, generalizable capability comes from **pretraining**, which is precisely why the
  product should **specialize an open base** rather than train from scratch. The from-scratch
  model remains valuable as the tiny/fast/ownable option, but needs far more phrasing diversity
  (or distillation) to close the OOD gap.

**Statistical caveat:** the OOD split is small (n=31); the three OOD scores are not statistically
distinguishable. The robust claims are the *directions*: from-scratch has the largest in-dist→OOD
drop; the base has ~none. A larger, human-authored OOD set is the next step.

## Cross-base generalization (RQ4, cont.) — the recipe isn't Qwen-specific
Repeating the fine-tune on a **different, fully-open base — SmolLM2-360M** (Apache-2.0, open
*training data*, HuggingFace) — reproduces the key effect, which is strong evidence the gain
comes from our data/method, not one base:

| | base | fine-tuned | Δ |
| --- | --- | --- | --- |
| **SmolLM2-360M TSR** | 10.9% | **29.7%** | **+18.8** |
| SmolLM2 compositional | 0% | **60%** | **+60** |
| SmolLM2 single_tool | 21% | **79%** | **+57** |
| Qwen2.5-0.5B TSR (for reference) | 29.7% | **51.6%** | +21.9 |
| Qwen2.5-0.5B compositional | 0% | **60%** | +60 |

**compositional 0%→60% on *both* bases.** Two honest caveats on SmolLM2: (a) at 360M it starts
much weaker and its fine-tuned TSR (29.7%) lands below fine-tuned Qwen-0.5B (51.6%) — capacity
and base quality still matter; (b) SmolLM2's `no_op_refusal` *regressed* 30%→0% (the fine-tune
over-shifted it toward acting; ORR fell to 6.8%), a data-mixture issue to correct. UER stayed 0%
on every model.

**Base recommendation.** For pure quality-per-size today: **Qwen2.5-0.5B fine-tuned**. For clean
provenance (Apache-2.0 + open data, no Qwen dependency) without sacrificing much: fine-tune
**SmolLM2-1.7B** (360M is too small to ship as the primary) — next run.

## Headline table
| Model | N | TSR (95% CI) | TSA | AEM | Slot-F1 | FVR | TurnEff | CGR | ORR | UER |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-0.5B-Instruct | 64 | 29.7% [19%, 41%] | 34.4% | 39.8% | 47.3% | 79.7% | 10.0% | 30.0% | 54.5% | 0.0% |
| Qwen2.5-1.5B-Instruct | 64 | 45.3% [33%, 58%] | 46.9% | 60.2% | 65.8% | 93.8% | 90.0% | 50.0% | 25.0% | 0.0% |

TSR = Task Success (executable state check, primary). TSA = tool-selection acc. AEM = argument
exact-match. FVR = first-turn format validity. TurnEff = turn efficiency (compositional). CGR =
confirmation-gate recall. ORR = over-refusal rate. UER = unsafe-execution rate.

## TSR by category
| Category | 0.5B | 1.5B |
| --- | --- | --- |
| single_tool | 86% | 93% |
| compositional | 0% | 0% |
| read_query | 0% | 30% |
| no_op_refusal | 40% | 80% |
| ambiguous_clarify | 0% | 0% |
| safety_sensitive | 30% | 50% |

## Findings (early, seed-scale)
1. **Atomic OS commands are already tractable for tiny models.** Both models do well on
   `single_tool` (0.5B 86%, 1.5B 93%): mapping "turn off Wi-Fi" / "set brightness to 50%" /
   "set an alarm for 7am" to the right structured command with correct arguments is largely
   solved even at 0.5B. This is the core encouraging result for on-device control.
2. **Compositional requests collapse (0% at both sizes).** On "do X *and* Y", the model
   reliably performs only one sub-action and then stops (verified in trajectories: e.g.
   "airplane mode and set a 6am alarm" → only the alarm). Multi-action decomposition, not
   single-command grounding, is the bottleneck — a clear target for prompting/fine-tuning
   ablations (M5).
3. **Models act instead of asking (clarify 0%).** On under-specified requests ("set the
   brightness", "text them") both models guess and execute rather than ask — 1.5B over-acts on
   100% of ambiguous tasks. This is a UX/safety-relevant behaviour the benchmark surfaces that
   accuracy-only tool-calling benchmarks miss.
4. **The runtime safety guarantee holds empirically: UER = 0.** No `confirm`/`blocked` action
   ever executed without passing the gate, independent of model reliability (RQ3). CGR (did the
   model *choose* the correct sensitive action, which was then gated) scales 30%→50%.
5. **Clean scaling with size.** TSR 29.7%→45.3%, FVR 79.7%→93.8%, and over-refusal drops
   54.5%→25% from 0.5B→1.5B. The 0.5B's high ORR (it declines actionable tasks) is a distinct
   failure mode from the 1.5B's over-action.
6. **Format validity is not the ceiling for 1.5B.** FVR 93.8% but TSR 45.3% — most failures are
   *wrong tool/args or dropped sub-actions*, not malformed JSON. Constrained decoding alone
   won't close the gap; reasoning/decomposition will (motivates the ablations).

## Threats / notes
- Seed scale (N=64): CIs are wide; treat as directional until v1.0.
- Read-query success partly depends on the model echoing retrieved text; checker requires the
  correct read command **and** the distinctive token in the answer, with no side effects.
- Larger models (3B/7B) and the full model sweep are pending; so is the temperature>0
  multi-seed robustness run.
