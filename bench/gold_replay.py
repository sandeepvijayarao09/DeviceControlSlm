"""Gold-replay validation: every task must pass its own checker under the
reference behaviour. A task whose gold annotation fails its checker is a broken
task and is rejected (benchmark-design.md §5). Run in CI.

    python -m bench.gold_replay
"""
from __future__ import annotations

import json
import sys
from typing import Any, Dict, List

from agentcore.backends import MockBackend

from .dataset import dataset_hash, load_tasks
from .runner import run_task


def _gold_script(task: Dict[str, Any]) -> List[str]:
    """Turn the gold `reference` into a scripted backend transcript that a
    correct agent would produce."""
    ctype = task["check"]["type"]
    turns: List[str] = []
    for call in task["reference"]["expected_calls"]:
        turns.append(json.dumps({"thought": "gold", "action": call}))

    if ctype == "refusal":
        turns.append(json.dumps({"thought": "gold", "final_answer":
                                 "I can't do that with the tools available on this device."}))
    elif ctype == "clarify":
        turns.append(json.dumps({"thought": "gold", "final_answer":
                                 "I can help — could you tell me the details?"}))
    elif ctype == "read":
        token = task["check"].get("answer_contains", "")
        turns.append(json.dumps({"thought": "gold", "final_answer": f"Here it is: {token}."}))
    else:
        turns.append(json.dumps({"thought": "gold", "final_answer": "Done."}))
    return turns


def validate(tasks: List[Dict[str, Any]] | None = None):
    """Return a list of (task_id, check_type, detail) for tasks whose gold
    behaviour fails their own checker. Empty list == all consistent."""
    tasks = tasks if tasks is not None else load_tasks()
    failures = []
    for task in tasks:
        backend = MockBackend(scripted=_gold_script(task))
        run = run_task(task, backend)
        if not run.success:
            failures.append((task["id"], task["check"]["type"], run.detail))
    return failures


def main() -> int:
    import sys
    tasks_dir = None
    if "--tasks-dir" in sys.argv:
        tasks_dir = sys.argv[sys.argv.index("--tasks-dir") + 1]
    tasks = load_tasks(tasks_dir=tasks_dir)
    failures = validate(tasks)
    print(f"gold replay: {len(tasks) - len(failures)}/{len(tasks)} tasks consistent")
    print(f"dataset hash: {dataset_hash(tasks)}")
    for tid, ctype, detail in failures:
        print(f"  FAIL {tid} ({ctype}): {detail}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
