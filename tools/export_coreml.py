"""Export the from-scratch device SLM to Core ML for on-device inference.

The model is tiny and char-level, so we export the *forward pass* (tokens ->
logits) as a fixed-shape Core ML program and run the greedy decode loop in Swift
(the char tokenizer + delimiters travel alongside as JSON). Fixed shape
[1, block_size] is the most portable option: on device, right-pad the token
window to block_size and read logits at the last real position.

    pip install coremltools torch
    python tools/export_coreml.py --ckpt dcslm/out/ckpt.pt --out apple/DeviceControlKit/Models

Conversion runs on Linux or macOS; the resulting .mlpackage runs on the Neural
Engine / GPU on iPhone, iPad, and Mac.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from scratch.model import GPT, GPTConfig, set_export_mode


class ForwardOnly(nn.Module):
    """Wrap GPT so the traced graph returns just logits for a fixed-length input."""
    def __init__(self, gpt: GPT):
        super().__init__()
        self.gpt = gpt

    def forward(self, tokens):                      # tokens: (1, block_size) int
        logits, _ = self.gpt(tokens)
        return logits                                # (1, block_size, vocab)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="dcslm/out/ckpt.pt")
    ap.add_argument("--out", default="apple/DeviceControlKit/Models")
    args = ap.parse_args(argv)

    ck = torch.load(Path(args.ckpt), map_location="cpu", weights_only=False)
    cfg = GPTConfig(**ck["cfg"])
    model = GPT(cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    set_export_mode(True)                            # Core ML-friendly attention/rope
    wrapper = ForwardOnly(model).eval()

    block = cfg.block_size
    example = torch.zeros(1, block, dtype=torch.long)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example)

    try:
        import coremltools as ct
    except ImportError:
        print("coremltools not installed — run `pip install coremltools` to produce the .mlpackage.")
        print("The tracing step above succeeded, so the model is export-ready.")
        return 0

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="tokens", shape=(1, block), dtype=int)],
        outputs=[ct.TensorType(name="logits")],
        minimum_deployment_target=ct.target.iOS16,
        compute_units=ct.ComputeUnit.ALL,          # Neural Engine + GPU + CPU
        convert_to="mlprogram",
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pkg = out / "DCSLM.mlpackage"
    mlmodel.save(str(pkg))

    # tokenizer + config for the Swift decode loop
    (out / "dcslm_tokenizer.json").write_text(json.dumps({
        "stoi": ck["stoi"],
        "itos": {str(k): v for k, v in ck["itos"].items()},
        "block_size": block,
        "vocab_size": cfg.vocab_size,
        "delimiters": {"user": "", "cmd": "", "end": ""},
    }, ensure_ascii=False), encoding="utf-8")

    size_mb = sum(f.stat().st_size for f in pkg.rglob("*")) / 1e6
    print(f"wrote {pkg} ({size_mb:.1f} MB) + dcslm_tokenizer.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
