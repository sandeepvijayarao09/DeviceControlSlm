"""Load the benchmark tasks and compute a version hash for reproducibility."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

TASKS_DIR = Path(__file__).resolve().parent / "tasks"

CATEGORY_ORDER = [
    "single_tool", "compositional", "read_query",
    "no_op_refusal", "ambiguous_clarify", "safety_sensitive",
]


def load_tasks(categories: List[str] | None = None,
               tasks_dir: Path | str | None = None) -> List[Dict[str, Any]]:
    base = Path(tasks_dir) if tasks_dir is not None else TASKS_DIR
    tasks: List[Dict[str, Any]] = []
    for cat in CATEGORY_ORDER:
        if categories is not None and cat not in categories:
            continue
        path = base / f"{cat}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                tasks.append(json.loads(line))
    return tasks


def dataset_hash(tasks: List[Dict[str, Any]]) -> str:
    """Stable content hash of the task set (order-independent by id)."""
    blob = json.dumps(sorted(tasks, key=lambda t: t["id"]), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
