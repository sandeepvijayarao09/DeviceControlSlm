# Model Card — DCSLM (device-control SLM)

Following Mitchell et al., "Model Cards for Model Reporting" (2019).

## Model details
- **Name:** DCSLM (Device-Control Small Language Model).
- **Architecture:** decoder-only transformer built from scratch (`scratch/model.py`) — RMSNorm,
  RoPE, grouped-query attention (8 query / 2 KV heads), SwiGLU, tied embeddings.
- **Size:** ~4.1M parameters; 6 layers × 256 dim; char-level tokenizer (~75 chars); context 160.
- **Training:** next-token prediction from random initialization on a synthetic device-control
  corpus; ~5k iterations on CPU. No pretrained weights (see `provenance.md`).
- **Dependency:** PyTorch (training) / Core ML (on-device inference).
- **Version / owner:** research prototype, this repository.
- **License:** intended permissive (code); the model is trained solely on our own synthetic data.

## Intended use
- **Primary:** map a natural-language request to a structured device-control command (or ASK /
  REFUSE) on iOS / iPadOS / macOS, executed via App Intents / Shortcuts / Accessibility under a
  confirmation gate.
- **Users:** the on-device agent runtime (`DeviceControlKit`), not end users directly.
- **Out of scope:** general conversation, world knowledge, non-device tasks, languages other than
  English, safety-critical automation without user confirmation.

## Factors
- Phrasing style (in-distribution vs. novel), request complexity (single vs. compositional),
  category (action / read / clarify / refuse / safety), and target OS (capability differs).

## Metrics (DeviceControlBench, executable TSR, temperature 0)
| Split | TSR (95% CI) | UER |
| --- | --- | --- |
| In-distribution seed (n=64) | 67.2% | 0% |
| **Held-out OOD (n=55)** | **32.7% [20, 46]** | 0% |

- **UER = 0%** (no sensitive action executed without passing the confirmation gate) — a runtime
  guarantee, independent of model accuracy.
- Honest comparison: a fine-tuned **pretrained** base (Qwen2.5-0.5B) scores lower in-distribution
  (51.6%) but **higher OOD (36.4%)** — pretraining generalizes better to novel phrasing.

## Evaluation data
`bench/tasks` (in-distribution) and `bench/tasks_ood` (independently authored, novel phrasing;
verified zero leakage vs. training via `tools/check_contamination.py`).

## Training data
~40k synthetic (request → DSL) examples from `dcslm/gen_corpus.py`: diverse phrasings + indirect
intents across the command catalog. Fully synthetic, no PII, no scraped data.

## Ethical considerations
- **Safety:** irreversible/outbound commands (restart, shutdown, message, mail, call) require
  explicit user confirmation; the gate cannot be bypassed by the model.
- **Privacy:** no personal data in training; on-device inference keeps requests local.
- **Failure modes:** may mis-map novel phrasing (see OOD result) or over-act on ambiguous
  requests — mitigated by clarify behavior and the confirmation gate.

## Caveats & recommendations
- The OOD gap means this from-scratch model is best where phrasing is guided/constrained; for
  open free-text robustness, prefer the fine-tuned open-base variant.
- `read_query` returns 0% (the model selects the query tool but the runtime must supply the value).
- English-only; small held-out set (n=55) — treat OOD numbers as directional.
