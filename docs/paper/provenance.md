# Provenance — what is "from scratch"

This project contains two model tracks. This document states, unambiguously, what is built from
scratch and what is not.

## DCSLM — the device-control model: **built from scratch**
The primary model (`scratch/model.py` + `dcslm/`) is from scratch in the standard ML sense:

- **Architecture written by us** — the transformer (causal attention, RMSNorm, RoPE,
  grouped-query attention, SwiGLU, tied embeddings) is our own code, not lifted from a pretrained
  model.
- **Weights trained from random initialization** — no pretraining, no fine-tuning of anyone's
  checkpoint. The ~4M parameters have never seen an external model.
- **Tokenizer built by us** — char-level, no BPE library, no borrowed vocabulary.
- **Training data synthesized by us** — `dcslm/gen_corpus.py`, not scraped or borrowed.

**The one dependency:** PyTorch, used for tensors, autodiff/backprop, standard layers
(`nn.Linear`, `nn.Embedding`, `softmax`) and the `AdamW` optimizer. We did not reimplement the
deep-learning framework itself — same sense in which nanoGPT and every "from-scratch" model are
from scratch. Rewriting PyTorch would not change the model or its results.

## Fine-tuned Qwen / SmolLM2 — **NOT from scratch**
`finetune/` specializes **pretrained, third-party** open models (Qwen2.5, SmolLM2) via LoRA.
These are **baselines and a possible alternative product path**, never presented as our own model.
They are kept because the held-out OOD test showed pretraining generalizes better to novel
phrasing (`results.md`), which is an honest, decision-relevant comparison.

## Summary
| Component | From scratch? |
| --- | --- |
| DCSLM architecture, tokenizer, weights, training data | ✅ yes |
| DCSLM's use of PyTorch (autodiff, layers, optimizer) | dependency, not reimplemented |
| Qwen / SmolLM2 fine-tunes | ❌ no — pretrained bases we specialized |
| Benchmark, agent runtime, checkers, App Intents, recipe engine | ✅ ours |
