# DCSLM — architecture & design of the from-scratch device-control SLM

A small language model whose **only** job is device control on **iOS / iPadOS / macOS**, built
**from scratch** (our own tokenizer, model, training, runtime — no pretrained base). This
document is the design of record.

## 1. Design thesis
General assistants are large because they must know everything. We need the opposite: a model
that does **one narrow thing**. That narrowness is the whole design lever:

- The **output space is a small closed set** (~40 commands) → we don't need a giant vocabulary
  or in-prompt tool schemas; the model *memorizes* the command set during training.
- The only thing left to learn is **natural language → command**, so a tiny network suffices.
- The risk shifts from "capacity" to "coverage": a from-scratch model has no borrowed language
  knowledge, so **its competence equals the diversity of our training data**. Data, not
  parameters, is the primary investment.

## 2. System architecture (one model, one runtime, three execution adapters)

```
   user text
      │
      ▼
┌───────────────────────────────────────────────┐
│ 2a. DCSLM model  (from-scratch GPT, ~4M params)│   emits a compact DSL:
│     char-level in → command DSL out            │   "set_wifi|enabled=false ; set_brightness|level=30"
└───────────────────────────────────────────────┘   or "ASK"  or  "REFUSE"
      │
      ▼
┌───────────────────────────────────────────────┐
│ 2b. Agent runtime  (shared, Swift on device)   │   parse DSL → validate → SAFETY GATE
│     plan-once, replay one command per turn      │   (safe / confirm / blocked) → execute
└───────────────────────────────────────────────┘
      │
      ▼
┌───────────────────────────────────────────────┐
│ 2c. Execution adapters  (per-OS)               │
│   iOS/iPadOS: App Intents, Shortcuts, EventKit │   ← restricted surface
│   macOS:      AppleScript, CoreWLAN, pmset, …  │   ← broad surface
└───────────────────────────────────────────────┘
```

The three layers are decoupled on purpose: the **model** is platform-agnostic (it outputs
*intent*), and *what a given OS can actually do* lives in the per-OS adapters + a capability
matrix — not in the model.

## 3. The model (from-scratch GPT)
Decoder-only transformer, **our code** (`scratch/model.py`), deliberately the same architecture
family as modern LLMs so it is honest infrastructure, just tiny:

| Property | Value |
| --- | --- |
| Type | decoder-only (causal) transformer |
| Params | ~4.1M (6 layers × 256 wide) |
| Heads | 8 query / 2 KV (**grouped-query attention**) |
| Norm / positions | **RMSNorm** (pre-norm) / **RoPE** |
| MLP | **SwiGLU** |
| Output head | tied to embedding |
| Tokenizer | **char-level** (~75 chars) — no BPE, fully ours, robust to typos/OOV |
| Context | 160 chars (a request + its command fit easily) |

Why char-level: the input is short and the vocabulary of a device request is small; char-level
needs zero external tokenizer, can never emit an out-of-vocab token, and degrades gracefully on
typos ("brihgtness"). The cost — spelling out words — is cheap at this sequence length.

## 4. The I/O contract (compact DSL)
Instead of verbose JSON tool-calls (used by the large baselines), the from-scratch model speaks
a compact DSL that is far easier for a tiny net to emit exactly:

```
request  →  tool|key=value|key=value [ ; tool|key=value … ]     # one or more commands
request  →  ASK                                                  # under-specified → clarify
request  →  REFUSE                                               # out of scope
```
Examples:
```
"turn off wifi and dim to 30%"      → set_wifi|enabled=false ; set_brightness|level=30
"set an alarm for 7am"              → set_alarm|time=07:00
"set the brightness"                → ASK
"transfer $500 to my landlord"      → REFUSE
```
Corpus lines are framed with single control chars `\x01request\x02dsl\x03` so a char-LM learns
example boundaries. The **same** framing is used at inference.

## 5. Training design
- **Objective:** next-token prediction (the same objective that trains every LLM), on the
  request→DSL corpus. The corpus *is* the pretraining — there is no separate general-text phase.
- **Corpus** (`dcslm/gen_corpus.py`): ~16k examples, ~13k unique prompts, heavy phrasing
  diversity per command (synonyms, politeness, word order, capitalization). Mixture ≈ 35%
  compositional / 43% single / 8% clarify / 8% refuse / 4% read.
- **Why diversity dominates:** the model can only answer phrasings whose *shape* it has seen.
  Every additional surface form widens the in-distribution cone. This is the main knob for
  quality and the main limitation (see §8).
- **Hardware:** trains on CPU in ~30 min (char-level, short sequences).

## 6. Runtime / inference design
`dcslm/backend.py` bridges the one-shot DSL to the turn-based agent loop:
1. On the first turn, **plan once**: greedy-decode the DSL from `\x01request\x02`, stop at
   `\x03`, parse into `(kind, actions)`.
2. Replay: emit **one command per turn** as the loop expects; the sandbox returns an
   observation each time; when the plan is exhausted, emit a final answer. `ASK`→a clarifying
   question, `REFUSE`→a decline.
3. Because it plugs in behind the standard `ModelBackend` interface, it inherits the **safety
   gate** (sensitive commands require confirmation) and is scored by the **same checkers** as
   every other model — so its number is directly comparable.

## 7. Evaluation design
Evaluated on **DeviceControlBench** unchanged, via `--backend scratch`. Success is executable
device state (not string match), with bootstrap 95% CIs and UER (unsafe-execution) tracked. This
keeps the from-scratch model honest against the specialized-base baselines on one ruler.

## 8. Honest limitations (by construction)
- **Generalization = data coverage.** Novel phrasings far from the corpus will fail; there is no
  world knowledge to fall back on. Mitigation: keep widening the corpus; optionally distill from
  a larger model's paraphrases.
- **Reads/queries are weak.** The model can pick `get_battery`, but it cannot *know* the value
  to speak — read tasks need the runtime to fill the answer.
- **Capacity ceiling.** At ~4M params it learns mappings, not reasoning; genuinely ambiguous or
  multi-constraint requests are hard.

## 9. Cross-platform deployment plan (next)
- **Export:** convert the trained weights to **Core ML** (`.mlpackage`, runs on the Neural
  Engine) and/or **MLX**; 4-bit quantize. One artifact for iPhone, iPad, Mac.
- **Runtime:** port the `agentcore` loop + `dcslm` planner to a **shared Swift package**.
- **Adapters + capability matrix:** implement `set_wifi`, `open_app`, … per OS (App Intents on
  iOS/iPadOS, AppleScript/system frameworks on macOS); annotate each command with per-platform
  availability so unsupported actions degrade gracefully.
