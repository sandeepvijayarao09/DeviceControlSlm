"""Leakage gate: assert no benchmark/test prompt appears in any training corpus.

Checks both splits against both training corpora:
  * test prompts  = bench/tasks (in-distribution seed) + bench/tasks_ood (held-out)
  * training text = dcslm/data/corpus.txt  +  finetune/data/{train,valid}.jsonl

Exits non-zero if any exact (normalized) test prompt is found in training — so it
can run as a CI gate. This is the standard train/test-separation guarantee.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

from bench.dataset import load_tasks  # noqa: E402


def norm(s: str) -> str:
    return " ".join(s.lower().split()).strip(" .!,?")


def test_prompts():
    prompts = {}
    for name, d in [("seed", ROOT / "bench" / "tasks"), ("ood", ROOT / "bench" / "tasks_ood")]:
        for t in load_tasks(tasks_dir=d):
            prompts[norm(t["prompt"])] = f"{name}:{t['id']}"
            for p in t.get("paraphrases", []):
                prompts[norm(p)] = f"{name}:{t['id']}(para)"
    return prompts


def training_prompts():
    train = set()
    corpus = ROOT / "dcslm" / "data" / "corpus.txt"
    if corpus.exists():
        for line in corpus.read_text(encoding="utf-8").splitlines():
            if "\x01" in line and "\x02" in line:
                train.add(norm(line[line.index("\x01") + 1: line.index("\x02")]))
    for split in ["train", "valid"]:
        p = ROOT / "finetune" / "data" / f"{split}.jsonl"
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                msgs = json.loads(line)["messages"]
                user = next((m["content"] for m in msgs if m["role"] == "user"), "")
                train.add(norm(user))
    return train


def main() -> int:
    tests = test_prompts()
    train = training_prompts()
    if not train:
        print("WARNING: no training corpora found (run `make ft-data` / `dcslm.gen_corpus`) "
              "— cannot verify leakage.")
        return 0
    leaks = {p: src for p, src in tests.items() if p in train}
    print(f"test prompts: {len(tests)} | training prompts: {len(train)}")
    print(f"LEAKS (test prompt found verbatim in training): {len(leaks)}")
    for p, src in list(leaks.items())[:20]:
        print(f"  - [{src}] {p!r}")
    if leaks:
        print("FAIL: train/test separation violated.")
        return 1
    print("OK: no test prompt appears in training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
