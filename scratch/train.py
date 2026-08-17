"""Pretrain the from-scratch GPT on a plain-text corpus (char-level).

This is real pretraining from RANDOM weights — the same objective (next-token
prediction) that produced Qwen, just tiny and on ~1 MB of text instead of ~18
trillion tokens. You can watch the loss fall and the samples turn from noise
into Shakespeare-ish text.

    python -m scratch.train --data scratch/data/tinyshakespeare.txt --iters 3000
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from .model import GPT, GPTConfig


def load_corpus(path: Path):
    text = path.read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    return data, stoi, itos


def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i: i + block_size] for i in ix])
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss(model, splits, block_size, batch_size, iters=20):
    model.eval()
    out = {}
    for name, d in splits.items():
        losses = torch.zeros(iters)
        for k in range(iters):
            x, y = get_batch(d, block_size, batch_size)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="scratch/data/tinyshakespeare.txt")
    ap.add_argument("--out", default="scratch/out")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--n-layer", type=int, default=6)
    ap.add_argument("--n-head", type=int, default=6)
    ap.add_argument("--n-kv-head", type=int, default=2)
    ap.add_argument("--n-embd", type=int, default=192)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)

    data, stoi, itos = load_corpus(Path(args.data))
    n = int(0.9 * len(data))
    splits = {"train": data[:n], "val": data[n:]}
    print(f"corpus: {len(data)} tokens | vocab: {len(stoi)} chars", flush=True)

    cfg = GPTConfig(vocab_size=len(stoi), block_size=args.block_size, n_layer=args.n_layer,
                    n_head=args.n_head, n_kv_head=args.n_kv_head, n_embd=args.n_embd)
    model = GPT(cfg)
    print(f"model: {model.num_params()/1e6:.2f}M non-embedding params "
          f"({sum(p.numel() for p in model.parameters())/1e6:.2f}M total)", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.iters)

    t0 = time.monotonic()
    for it in range(1, args.iters + 1):
        x, y = get_batch(splits["train"], args.block_size, args.batch_size)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if it % args.eval_every == 0 or it == 1:
            m = estimate_loss(model, splits, args.block_size, args.batch_size)
            rate = it / (time.monotonic() - t0)
            print(f"iter {it:5d} | train {m['train']:.3f} | val {m['val']:.3f} "
                  f"| {rate:.1f} it/s", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "stoi": stoi, "itos": itos},
               out / "ckpt.pt")
    print(f"saved checkpoint to {out/'ckpt.pt'}", flush=True)

    # show a sample so you can see what it learned
    ctx = torch.zeros((1, 1), dtype=torch.long)
    sample = model.generate(ctx, max_new_tokens=400, temperature=0.8, top_k=40)[0].tolist()
    print("\n----- sample -----\n" + "".join(itos[i] for i in sample), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
