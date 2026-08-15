# Fine-tuning the device-control SLM

This is where we **build our SLM**: we don't pretrain from scratch — we take a strong open
Apache-2.0 base and specialize it for structured device control via **LoRA**, targeting the
exact failures the base models show on DeviceControlBench (compositional collapse; acting
instead of asking). See `docs/paper/research-plan.md` §M5.

## Why fine-tune (and why these bases)
The seed baselines (`results/leaderboard.md`) show ≤4B open models already handle *atomic*
commands but fail at *chaining* and *clarifying*. Fine-tuning on trajectories that demonstrate
those behaviours is the direct fix. Candidate bases are all **Apache-2.0** and MLX-ready, so
the result is shippable in a commercial Apple app:
- `Qwen2.5-0.5B-Instruct` — smallest, best for phone RAM/battery
- `Qwen2.5-1.5B-Instruct` — strongest baseline so far
- `SmolLM2-1.7B-Instruct` — fully open (data included)

We fine-tune all three and let the benchmark pick the winner.

## The training data (built here, no GPU needed)
`gen_data.py` synthesizes SFT trajectories in **MLX chat format** (`{"messages": [...]}`),
using the *same* system prompt + tool rendering as inference (zero train/inference skew), and
**rolls out gold trajectories through the sandbox** so multi-step examples teach act→observe→act
→finish. It is **contamination-controlled**: any prompt colliding with a DeviceControlBench
task/paraphrase is dropped.

```bash
make ft-data          # -> finetune/data/{train,valid}.jsonl  (~2400 examples)
```
Composition (approx): ~49% compositional (multi-step), ~27% single/read/comms, ~14% refuse,
~10% clarify. Every gold action turn is verified to parse and validate against the tool schema.

## Two training paths
| Path | Script | Runs on | Use |
| --- | --- | --- | --- |
| **Portable (HF + PEFT)** | `train_hf.py` | any CPU/GPU, any OS | reproducible fine-tune + eval anywhere (incl. CI); no Apple hardware needed |
| **On-device (MLX)** | `train_and_eval.sh` | Apple Silicon | the shipping path — trains the model we deploy, then measures on-device latency/memory/energy |

Both consume the *same* `finetune/data` and are evaluated by the *same* benchmark, so results
are comparable. The portable path lets us build and validate the specialized model without a
Mac; the MLX path produces the quantized on-device artifact.

### Portable path (CPU/GPU, any OS)
```bash
pip install torch transformers peft datasets
python -m finetune.train_hf --model Qwen/Qwen2.5-0.5B-Instruct \
    --out finetune/merged/qwen0.5b-ft --max-examples 2500 --max-seq-len 1200
python -m bench.harness --backend hf --model finetune/merged/qwen0.5b-ft --tag ft
python -m bench.summarize        # base vs ft, side by side
```
Each chat example is expanded into per-assistant-turn `(prompt, completion)` pairs with the
prompt masked, so every decision — including the *later* steps of a compositional trajectory —
is supervised. `--max-seq-len 1200` fits every example (p99≈1072 tok), so tool definitions are
never truncated out of the prompt.

## Train + evaluate (on an Apple Silicon Mac)
```bash
pip install "mlx-lm>=0.19"
bash finetune/train_and_eval.sh      # BEFORE vs AFTER benchmark for each base
```
This, per base: runs the zero-shot benchmark (`--tag base`), LoRA-fine-tunes
(`mlx_lm lora`, config in `lora_config.yaml`), fuses the adapter into a standalone model
(`mlx_lm fuse`), re-runs the benchmark (`--tag ft`), and finally aggregates a leaderboard.

Manual, single base:
```bash
python -m mlx_lm lora --model mlx-community/Qwen2.5-1.5B-Instruct-4bit --train \
  --data finetune/data --adapter-path finetune/adapters/qwen2.5-1.5b \
  --iters 800 --batch-size 4 --num-layers 16 --max-seq-length 2048 --grad-checkpoint
python -m mlx_lm fuse  --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
  --adapter-path finetune/adapters/qwen2.5-1.5b --save-path finetune/fused/qwen2.5-1.5b
python -m bench.harness --backend mlx --model finetune/fused/qwen2.5-1.5b --tag ft
```

## What "success" looks like
A fine-tuned model with materially higher **TSR** driven by the previously-zero categories —
**compositional** and **ambiguous_clarify** — while keeping **UER = 0** (the runtime guarantee
is independent of the model). The winner (best TSR at acceptable size/latency on your Mac)
becomes our shipped SLM; we then quantize for on-device and measure MLX latency/memory/energy.

## Notes
- Adapters/fused weights are **git-ignored** (large, machine-specific). Commit *metrics*
  (`results/`), not weights.
- `lora_config.yaml` holds the shared hyperparameters; the script overrides `model`/paths.
- Trajectories cap at 2048 tokens; raise `--max-seq-length` if you enlarge the tool set.
