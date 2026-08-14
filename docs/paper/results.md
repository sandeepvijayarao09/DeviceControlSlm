# Results (living) — DeviceControlBench v0.2 seed

First real runs on the reoriented OS-level benchmark. **Portable quality track** (HF
Transformers, CPU, greedy/temperature 0, deterministic). On-device efficiency numbers (MLX on
Apple Silicon) are pending. These are seed-set numbers (N=64); v1.0 (≥500) is milestone M6.

Dataset hash: `cf1ed9dcd23ebc0f` · reproduce with `make eval MODEL=... && make summarize`.

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
