"""Eval runner: executes the task suite against a backend and reports metrics.

Metrics (docs/design.md §8):
  * task success rate      - fraction of tasks whose `check` passes
  * tool-call validity rate - valid tool calls / total tool calls attempted
  * avg steps / task

Usage:
  python -m eval.runner                # rule-based reference policy (default)
  python -m eval.runner --backend mlx  # real model, on a Mac with mlx-lm
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import List

from agentcore import Agent, ToolRegistry
from tools import DeviceState, build_tools

from .reference_policy import reference_policy
from .tasks import TASKS, Task


@dataclass
class TaskOutcome:
    task_id: str
    success: bool
    steps: int
    tool_calls: int
    invalid_calls: int
    stopped_reason: str
    answer: str


def _make_backend(kind: str):
    if kind == "reference":
        from agentcore.backends import MockBackend

        return lambda: MockBackend(policy=reference_policy)
    if kind == "mlx":
        from agentcore.backends.mlx_backend import MLXBackend

        shared = MLXBackend()  # load once, reuse across tasks
        return lambda: shared
    raise SystemExit(f"unknown backend: {kind}")


def run_task(task: Task, make_backend) -> TaskOutcome:
    state = DeviceState(notes={"Home Wi-Fi": "wifi password is hunter2"})
    registry = ToolRegistry(build_tools(state))
    agent = Agent(make_backend(), registry)
    result = agent.run(task.request)

    invalid = sum(1 for s in result.steps if s.kind == "error")
    return TaskOutcome(
        task_id=task.id,
        success=bool(task.check(result, state)),
        steps=len(result.steps),
        tool_calls=len(result.tool_calls),
        invalid_calls=invalid,
        stopped_reason=result.stopped_reason,
        answer=result.answer or "",
    )


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the DeviceControlSlm eval suite.")
    parser.add_argument("--backend", default="reference", choices=["reference", "mlx"])
    args = parser.parse_args(argv)

    make_backend = _make_backend(args.backend)
    outcomes = [run_task(t, make_backend) for t in TASKS]

    passed = sum(o.success for o in outcomes)
    total = len(outcomes)
    total_calls = sum(o.tool_calls for o in outcomes)
    total_invalid = sum(o.invalid_calls for o in outcomes)
    valid_calls = total_calls  # tool_calls counts only validated calls
    attempted = total_calls + total_invalid
    validity = (valid_calls / attempted) if attempted else 1.0
    avg_steps = sum(o.steps for o in outcomes) / total if total else 0.0

    print(f"backend: {args.backend}")
    print(f"{'task':28} {'ok':>3} {'steps':>6} {'calls':>6} {'invalid':>8}  stop")
    for o in outcomes:
        mark = "PASS" if o.success else "FAIL"
        print(
            f"{o.task_id:28} {mark:>3} {o.steps:>6} {o.tool_calls:>6} "
            f"{o.invalid_calls:>8}  {o.stopped_reason}"
        )
    print("-" * 68)
    print(f"task success rate:       {passed}/{total} = {passed / total:.0%}")
    print(f"tool-call validity rate: {validity:.0%}  ({valid_calls}/{attempted} valid)")
    print(f"avg steps/task:          {avg_steps:.1f}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
