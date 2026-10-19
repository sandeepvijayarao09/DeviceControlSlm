"""Generate text from a from-scratch checkpoint.

    python -m scratch.sample --prompt "ROMEO:" --tokens 300
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .model import GPT, GPTConfig


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="scratch/out/ckpt.pt")
    ap.add_argument("--prompt", default="\n")
    ap.add_argument("--tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    ck = torch.load(Path(args.ckpt), map_location="cpu", weights_only=False)
    model = GPT(GPTConfig(**ck["cfg"]))
    model.load_state_dict(ck["model"])
    model.eval()
    stoi, itos = ck["stoi"], ck["itos"]

    idx = torch.tensor([[stoi.get(c, 0) for c in args.prompt]], dtype=torch.long)
    out = model.generate(idx, args.tokens, temperature=args.temperature, top_k=args.top_k)[0].tolist()
    print("".join(itos[i] for i in out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
