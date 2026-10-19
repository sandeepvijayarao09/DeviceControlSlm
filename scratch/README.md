# From-scratch transformer (learning + ownership track)

A complete transformer LM **built from random weights, entirely in this repo** — our own
tokenizer, attention, and training loop. No pretrained weights are loaded. PyTorch is used only
for tensors + autograd (the calculus); the *architecture* is ours (`model.py`).

It is deliberately a **miniature of our product base's architecture family (Qwen2.5)** so it
doubles as documentation of how that model works:

| Component | Here (`model.py`) | Qwen2.5-0.5B | Same idea? |
| --- | --- | --- | --- |
| Type | decoder-only transformer | decoder-only transformer | ✅ |
| Norm | `RMSNorm` (pre-norm) | RMSNorm (pre-norm) | ✅ |
| Positions | `apply_rope` (RoPE) | RoPE | ✅ |
| Attention | GQA (`n_head=6`, `n_kv_head=2`) | GQA (14 / 2) | ✅ |
| MLP | `SwiGLU` | SwiGLU | ✅ |
| Output head | tied to embedding | tied | ✅ |
| Layers × width | 6 × 192 | 24 × 896 | scaled down |
| Params | ~2.4M | ~490M | **~200× smaller** |
| Pretraining data | ~1 MB (TinyShakespeare) | **~18,000,000 MB** | **~18M× less** |

## Run it
```bash
make scratch-train      # pretrain ~2.4M params on CPU (~25 min); prints a sample
python -m scratch.sample --prompt "ROMEO:" --tokens 300
```
Watch the loss fall from ~4.0 (random: ln(65)≈4.17) toward ~1.5, and the samples turn from
noise into Shakespeare-shaped text (character names, line breaks, dialogue rhythm).

## Why this is NOT the product model (the lesson)
This model learns the *shape* of English from 1 MB of one author. It cannot understand an
arbitrary request like "I'm heading to bed" → Sleep Focus + alarm + DND, because that needs
broad world/language knowledge that only comes from pretraining on trillions of tokens. That is
exactly the capability we get "for free" by starting the **product** from an open base and
specializing it (`finetune/`).

The honest trade-off:

| | From scratch (this) | Specialize an open base (`finetune/`) |
| --- | --- | --- |
| Own every line / weight | ✅ | ✅ (Apache-2.0 fine-tune is legally ours) |
| Understand the whole stack | ✅ best for learning | partial |
| Language understanding | tiny (1 MB corpus) | strong (base pretraining) |
| Cost to reach product quality | ~1,000+ GPU-days + 18T curated tokens | ~45 min LoRA on CPU |
| Best shipped model | ✗ | ✅ |

**Conclusion:** build-from-scratch is the right tool for *learning* and for *owning the
architecture*; specialization is the right tool for *shipping the best on-device model*. We do
both — this directory for the first, `finetune/` for the second.

## Removing the Qwen dependency (ownership / provenance)
If the concern is depending on Qwen specifically, the product pipeline is base-agnostic: point
it at **SmolLM2** (HuggingFace, Apache-2.0, *open training data*) instead — same
`train_hf.py`, same benchmark. That keeps the shipping quality while making the base fully open
and Western-origin.
