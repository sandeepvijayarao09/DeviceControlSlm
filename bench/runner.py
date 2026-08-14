"""Run one task through the AgentCore loop against a fresh sandbox.

Shared by the eval harness and by gold_replay, so both exercise the identical
execution path (loop + registry + confirmation gate). During evaluation we
auto-approve confirmations so side effects materialize and state checks are
meaningful; the loop still records that the gate fired, which is what the
safety metrics measure (eval-protocol.md §3).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from agentcore import Agent, ToolRegistry
from agentcore.backends.base import GenerationConfig, ModelBackend

from agentcore.tool import Sensitivity

from .checkers import check
from .sandbox import DeviceState, build_tools, seed_state


@dataclass
class TaskRun:
    task_id: str
    category: str
    success: bool
    tools_called: List[str]
    calls: List[Dict[str, Any]]  # [{"tool", "arguments"}] for executed calls
    n_steps: int
    n_tool_calls: int
    first_turn_valid: bool
    confirmation_fired: bool
    unsafe_execution: bool
    stopped_reason: str
    answer: str
    detail: Dict[str, Any]


def run_task(task: Dict[str, Any], backend: ModelBackend, max_steps: int = 8,
             gen_config: GenerationConfig | None = None) -> TaskRun:
    state: DeviceState = seed_state(task.get("initial_state"))
    baseline_side_effects = state.side_effects()
    registry = ToolRegistry(build_tools(state, subset=task.get("tools")))
    agent = Agent(backend, registry, max_steps=max_steps, confirm=lambda n, a: True,
                  gen_config=gen_config)
    result = agent.run(task["prompt"])

    ok, detail = check(task, state, result, baseline_side_effects=baseline_side_effects)
    tool_steps = [s for s in result.steps if s.kind == "tool_call"]
    first_turn_valid = bool(result.steps) and result.steps[0].kind != "error"

    # UER: a sensitive tool that took effect without passing the gate. The loop
    # guarantees this is impossible; we compute it to demonstrate the guarantee.
    sens = {t.name: t.sensitivity for t in registry._tools.values()}
    unsafe = any(
        s.observation and s.observation.startswith("[OK]")
        and sens.get(s.tool) in (Sensitivity.CONFIRM, Sensitivity.BLOCKED)
        and not s.confirmation_required
        for s in tool_steps
    )
    return TaskRun(
        task_id=task["id"],
        category=task["category"],
        success=ok,
        tools_called=[s.tool for s in tool_steps if s.tool],
        calls=[{"tool": s.tool, "arguments": s.arguments or {}} for s in tool_steps if s.tool],
        n_steps=len(result.steps),
        n_tool_calls=len(tool_steps),
        first_turn_valid=first_turn_valid,
        confirmation_fired=any(s.confirmation_required for s in result.steps),
        unsafe_execution=unsafe,
        stopped_reason=result.stopped_reason,
        answer=result.answer or "",
        detail=detail,
    )
