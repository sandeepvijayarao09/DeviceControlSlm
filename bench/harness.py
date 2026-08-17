"""Evaluation harness: run a model over the benchmark, compute metrics with
bootstrap CIs, and write an auditable run.json (eval-protocol §8).

    python -m bench.harness --backend hf --model Qwen/Qwen2.5-0.5B-Instruct
    python -m bench.harness --backend mlx --model mlx-community/Qwen2.5-3B-Instruct-4bit  # on a Mac
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from agentcore.backends.base import GenerationConfig

from .dataset import dataset_hash, load_tasks
from .metrics import compute
from .runner import TaskRun, run_task
from .stats import bootstrap_ci

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def make_backend(kind: str, model: str, dtype: str):
    if kind == "hf":
        from .backends_hf import HFBackend
        return HFBackend(model, dtype=dtype)
    if kind == "mlx":
        from agentcore.backends.mlx_backend import MLXBackend
        return MLXBackend(model)
    if kind == "mock":  # rule-based reference policy, for pipeline smoke tests only
        from agentcore.backends import MockBackend
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
        from eval.reference_policy import reference_policy
        return MockBackend(policy=reference_policy)
    if kind == "scratch":  # our from-scratch, device-control-only SLM (ckpt path in `model`)
        from dcslm.backend import ScratchDeviceBackend
        return ScratchDeviceBackend(model)
    raise SystemExit(f"unknown backend: {kind}")


def run_eval(backend, tasks: List[Dict[str, Any]], max_tokens: int, temperature: float,
             max_steps: int, limit: int | None = None, verbose: bool = False) -> List[TaskRun]:
    cfg = GenerationConfig(max_tokens=max_tokens, temperature=temperature)
    runs: List[TaskRun] = []
    subset = tasks[:limit] if limit else tasks
    for i, task in enumerate(subset, 1):
        t0 = time.monotonic()
        run = run_task(task, backend, max_steps=max_steps, gen_config=cfg)
        if verbose:
            print(f"[{i}/{len(subset)}] {task['id']:16} "
                  f"{'PASS' if run.success else 'fail':4} "
                  f"tools={run.tools_called} ({time.monotonic()-t0:.1f}s)", flush=True)
        runs.append(run)
    return runs


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run DeviceControlBench.")
    ap.add_argument("--backend", default="hf", choices=["hf", "mlx", "mock", "scratch"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="run only first N tasks (smoke test)")
    ap.add_argument("--categories", nargs="*", default=None)
    ap.add_argument("--tag", default=None, help="results subdir label")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    tasks = load_tasks(args.categories)
    ds_hash = dataset_hash(load_tasks())
    print(f"loading backend={args.backend} model={args.model} ...")
    t_load = time.monotonic()
    backend = make_backend(args.backend, args.model, args.dtype)
    load_s = time.monotonic() - t_load

    print(f"running {len(tasks)} tasks (max_steps={args.max_steps}) ...")
    t0 = time.monotonic()
    runs = run_eval(backend, tasks, args.max_tokens, args.temperature, args.max_steps,
                    args.limit, args.verbose)
    wall = time.monotonic() - t0

    metrics = compute(load_tasks(args.categories), runs)
    tsr_mean, tsr_lo, tsr_hi = bootstrap_ci(metrics.pop("_tsr_vector"), seed=args.seed)

    manifest = {
        "model": args.model,
        "backend": args.backend,
        "revision": getattr(backend, "revision", "unknown"),
        "dtype": args.dtype,
        "temperature": args.temperature,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "max_steps": args.max_steps,
        "dataset_hash": ds_hash,
        "n_tasks": len(runs),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "load_seconds": round(load_s, 1),
        "wall_seconds": round(wall, 1),
    }
    record = {
        "manifest": manifest,
        "metrics": metrics,
        "TSR_ci95": [round(tsr_lo, 4), round(tsr_hi, 4)],
        "runs": [r.__dict__ for r in runs],
    }

    tag = args.tag or f"{args.temperature}T"
    out_dir = RESULTS_DIR / args.model.replace("/", "__") / f"{args.backend}_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run.json").write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")

    print("\n=== DeviceControlBench ===")
    print(f"model: {args.model}  backend: {args.backend}  n={len(runs)}  ({wall:.0f}s)")
    print(f"TSR (Task Success):      {metrics['TSR']:.1%}  95% CI [{tsr_lo:.1%}, {tsr_hi:.1%}]")
    print(f"TSA (Tool Selection):    {metrics['TSA']:.1%}")
    print(f"AEM (Arg Exact-Match):   {metrics['AEM']:.1%}")
    print(f"Slot-F1:                 {metrics['Slot_F1']:.1%}")
    print(f"FVR (Format Validity):   {metrics['FVR']:.1%}")
    print(f"Turn Efficiency:         {metrics['Turn_Efficiency']:.1%}")
    s = metrics["safety"]
    print(f"Safety  CGR={s['CGR']:.1%}  ORR={s['ORR']:.1%}  UER={s['UER']:.1%}  (n_confirm={s['n_confirm_tasks']})")
    print("by category:")
    for cat, v in metrics["TSR_by_category"].items():
        print(f"  {cat:20} {v:.1%}")
    print(f"\nwrote {out_dir / 'run.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
