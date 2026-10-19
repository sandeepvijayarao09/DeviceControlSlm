"""Portable LoRA fine-tuning (CPU/GPU, any OS) via HuggingFace + PEFT.

This is the backend-agnostic training path that mirrors our portable HF *eval*
backend: it runs anywhere (no Apple hardware needed), so the whole
build-our-SLM loop — generate data -> fine-tune -> evaluate -> compare — is
reproducible on a plain machine and in CI. The MLX path (`train_and_eval.sh`)
is the on-device/Apple-Silicon equivalent for shipping.

Each multi-turn chat example is expanded into per-assistant-turn
(prompt, completion) pairs with the prompt masked, so the model is supervised
on every decision — including the later steps of a compositional trajectory
(act -> observe -> ACT -> ... -> finish), which is exactly the base's weak spot.

    python -m finetune.train_hf --model Qwen/Qwen2.5-0.5B-Instruct \
        --data finetune/data --out finetune/merged/qwen0.5b --max-examples 2000
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_examples(tok, path: Path, max_seq_len: int, max_examples: int, seed: int):
    """Expand chat conversations into masked (input_ids, labels) training pairs."""
    rng = random.Random(seed)
    convs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    rng.shuffle(convs)
    eos = tok.eos_token_id
    examples = []
    for conv in convs:
        msgs = conv["messages"]
        for i, m in enumerate(msgs):
            if m["role"] != "assistant":
                continue
            prompt_text = tok.apply_chat_template(msgs[:i], add_generation_prompt=True, tokenize=False)
            p_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
            c_ids = tok(m["content"], add_special_tokens=False)["input_ids"] + [eos]
            # keep the whole completion; left-trim prompt if over budget
            budget = max_seq_len - len(c_ids)
            if budget < 1:
                c_ids = c_ids[-(max_seq_len - 1):]
                budget = max_seq_len - len(c_ids)
            p_ids = p_ids[-budget:]
            input_ids = p_ids + c_ids
            labels = [-100] * len(p_ids) + c_ids
            examples.append((input_ids, labels))
            if len(examples) >= max_examples:
                return examples
    return examples


def collate(batch, pad_id):
    maxlen = max(len(x[0]) for x in batch)
    input_ids, labels, attn = [], [], []
    for ids, lbl in batch:
        pad = maxlen - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        labels.append(lbl + [-100] * pad)
        attn.append([1] * len(ids) + [0] * pad)
    return (torch.tensor(input_ids), torch.tensor(labels), torch.tensor(attn))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--data", default="finetune/data")
    ap.add_argument("--out", required=True, help="output dir for the merged fine-tuned model")
    ap.add_argument("--max-examples", type=int, default=2000)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--report-every", type=int, default=10, help="optimizer steps between loss logs")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.config.use_cache = False

    lora = LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.train()

    examples = build_examples(tok, Path(args.data) / "train.jsonl",
                              args.max_seq_len, args.max_examples, args.seed)
    print(f"training examples (per-assistant-turn): {len(examples)}", flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    pad_id = tok.pad_token_id
    step, t0 = 0, time.monotonic()
    for epoch in range(args.epochs):
        random.Random(args.seed + epoch).shuffle(examples)
        opt.zero_grad()
        run_loss = 0.0
        for bi in range(0, len(examples), args.batch_size):
            batch = examples[bi: bi + args.batch_size]
            input_ids, labels, attn = collate(batch, pad_id)
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            loss = out.loss / args.grad_accum
            loss.backward()
            run_loss += out.loss.item()
            if (bi // args.batch_size + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                opt.zero_grad()
                step += 1
                if step % args.report_every == 0:
                    avg = run_loss / (args.grad_accum * args.report_every)
                    rate = (bi + args.batch_size) / (time.monotonic() - t0)
                    print(f"epoch {epoch} step {step} loss {avg:.3f} ({rate:.2f} ex/s)", flush=True)
                    run_loss = 0.0

    print("merging LoRA and saving merged model ...", flush=True)
    merged = model.merge_and_unload()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"saved merged model to {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
