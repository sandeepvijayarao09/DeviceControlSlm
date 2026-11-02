"""Aggregate all results/*/run.json into a leaderboard (Markdown + CSV).

Dependency-free so it runs anywhere. Figures for the paper are produced from
the same run.json files; this is the canonical, auditable summary.

    python -m bench.summarize
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

COLUMNS = ["TSR", "TSA", "AEM", "Slot_F1", "FVR", "Turn_Efficiency"]
SAFETY = ["CGR", "ORR", "UER"]
# Runs smaller than this are harness smoke tests, not results; their run.json
# stays on disk for the record but is left off the leaderboard.
MIN_TASKS = 10


def _load() -> List[Dict[str, Any]]:
    rows = []
    for rj in sorted(RESULTS_DIR.glob("*/*/run.json")):
        d = json.loads(rj.read_text(encoding="utf-8"))
        m, man = d["metrics"], d["manifest"]
        if man["n_tasks"] < MIN_TASKS:
            continue
        row = {
            "set": "held-out" if rj.parent.name.endswith("_ood") else "in-dist",
            "model": man["model"], "backend": man["backend"],
            "temp": man["temperature"], "n": man["n_tasks"],
            "TSR_ci95": d.get("TSR_ci95"),
        }
        row.update({c: m[c] for c in COLUMNS})
        row.update({c: m["safety"][c] for c in SAFETY})
        rows.append(row)
    return sorted(rows, key=lambda r: (r["set"] != "in-dist", -r["TSR"]))


def _fmt(x: float) -> str:
    return f"{x:.1%}"


def main() -> int:
    rows = _load()
    if not rows:
        print("no results yet — run `python -m bench.harness ...` first")
        return 0

    # Markdown leaderboard
    md = ["# DeviceControlBench — Leaderboard", "",
          "| Set | Model | Backend | T | N | TSR (95% CI) | TSA | AEM | Slot-F1 | FVR | TurnEff | CGR | ORR | UER |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        ci = r["TSR_ci95"]
        ci_s = f"{_fmt(r['TSR'])} [{ci[0]:.0%},{ci[1]:.0%}]" if ci else _fmt(r["TSR"])
        md.append("| " + " | ".join([
            r["set"], r["model"], r["backend"], str(r["temp"]), str(r["n"]), ci_s,
            _fmt(r["TSA"]), _fmt(r["AEM"]), _fmt(r["Slot_F1"]), _fmt(r["FVR"]),
            _fmt(r["Turn_Efficiency"]), _fmt(r["CGR"]), _fmt(r["ORR"]), _fmt(r["UER"]),
        ]) + " |")
    (RESULTS_DIR / "leaderboard.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # CSV
    with (RESULTS_DIR / "leaderboard.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["set", "model", "backend", "temp", "n"] + COLUMNS + SAFETY)
        for r in rows:
            w.writerow([r["set"], r["model"], r["backend"], r["temp"], r["n"]]
                       + [r[c] for c in COLUMNS] + [r[c] for c in SAFETY])

    print("\n".join(md))
    print(f"\nwrote {RESULTS_DIR/'leaderboard.md'} and leaderboard.csv")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
