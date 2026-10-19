"""The agent loop: the portable heart of the system (docs/design.md §4.1).

    build prompt -> generate -> parse -> validate -> [confirm] -> execute
    -> observe -> repeat, until final answer / step cap / time budget.

Kept free of any I/O or platform code so it ports cleanly to Swift AgentCore.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from .backends.base import GenerationConfig, Message, ModelBackend
from .parser import Decision, parse_decision
from .registry import ToolRegistry
from .tool import Sensitivity, ToolResult

_PROMPT_PATH = Path(__file__).resolve().parents[2] / "shared" / "prompts" / "system.md"

# Called before a CONFIRM-sensitivity tool runs. Return True to allow.
# In the app this is the confirmation sheet (ADR-0005); default auto-approves
# so headless eval can run, but the loop records that a confirmation was due.
ConfirmFn = Callable[[str, dict], bool]


@dataclass
class Step:
    kind: str  # "tool_call" | "final" | "error"
    thought: str = ""
    tool: Optional[str] = None
    arguments: Optional[dict] = None
    observation: Optional[str] = None
    confirmation_required: bool = False
    raw: str = ""


@dataclass
class AgentResult:
    answer: Optional[str]
    steps: List[Step] = field(default_factory=list)
    stopped_reason: str = "final_answer"  # or "max_steps" | "time_budget" | "blocked"

    @property
    def tool_calls(self) -> List[Step]:
        return [s for s in self.steps if s.kind == "tool_call"]


class Agent:
    def __init__(
        self,
        backend: ModelBackend,
        registry: ToolRegistry,
        max_steps: int = 6,
        time_budget_s: float = 60.0,
        confirm: Optional[ConfirmFn] = None,
        gen_config: Optional[GenerationConfig] = None,
        system_prompt_path: Path = _PROMPT_PATH,
    ) -> None:
        self.backend = backend
        self.registry = registry
        self.max_steps = max_steps
        self.time_budget_s = time_budget_s
        # Default: auto-approve so eval is non-interactive. The loop still marks
        # that a confirmation was due, so tests can assert the gate fired.
        self.confirm = confirm or (lambda name, args: True)
        self.gen_config = gen_config or GenerationConfig()
        self._template = system_prompt_path.read_text(encoding="utf-8")

    def _system_message(self) -> Message:
        content = self._template.replace("{tools}", self.registry.render_for_prompt())
        return {"role": "system", "content": content}

    def run(self, user_request: str) -> AgentResult:
        messages: List[Message] = [self._system_message(), {"role": "user", "content": user_request}]
        result = AgentResult(answer=None)
        start = time.monotonic()

        for _ in range(self.max_steps):
            if time.monotonic() - start > self.time_budget_s:
                result.stopped_reason = "time_budget"
                return result

            raw = self.backend.generate(messages, self.gen_config)
            decision = parse_decision(raw)

            if decision.parse_error and not decision.is_tool_call and not decision.is_final:
                step, done = self._handle_invalid(messages, raw, decision.parse_error)
                result.steps.append(step)
                if done:
                    result.stopped_reason = "error"
                    return result
                continue

            if decision.is_final:
                result.steps.append(Step(kind="final", thought=decision.thought, raw=raw))
                result.answer = decision.final_answer
                result.stopped_reason = "final_answer"
                return result

            # tool call
            assert decision.tool is not None and decision.arguments is not None
            ok, err = self.registry.validate(decision.tool, decision.arguments)
            if not ok:
                step, done = self._handle_invalid(messages, raw, f"tool call rejected: {err}")
                result.steps.append(step)
                if done:
                    result.stopped_reason = "error"
                    return result
                continue

            tool = self.registry.get(decision.tool)
            assert tool is not None
            step = Step(
                kind="tool_call",
                thought=decision.thought,
                tool=decision.tool,
                arguments=decision.arguments,
                raw=raw,
            )

            if tool.sensitivity is Sensitivity.BLOCKED:
                step.observation = "[ERROR] tool is blocked and cannot be used"
                self._record_turn(messages, raw, step.observation)
                result.steps.append(step)
                continue

            if tool.sensitivity is Sensitivity.CONFIRM:
                step.confirmation_required = True
                if not self.confirm(decision.tool, decision.arguments):
                    step.observation = "[ERROR] user declined the action"
                    self._record_turn(messages, raw, step.observation)
                    result.steps.append(step)
                    continue

            observation = self._execute(tool.invoke, decision.arguments)
            step.observation = observation.render()
            self._record_turn(messages, raw, step.observation)
            result.steps.append(step)

        result.stopped_reason = "max_steps"
        return result

    # --- helpers ----------------------------------------------------------
    def _execute(self, invoke: Callable[[dict], ToolResult], args: dict) -> ToolResult:
        try:
            return invoke(args)
        except Exception as e:  # tools must not crash the loop
            return ToolResult(ok=False, content=f"tool raised: {e}")

    def _record_turn(self, messages: List[Message], assistant_raw: str, observation: str) -> None:
        messages.append({"role": "assistant", "content": assistant_raw})
        messages.append({"role": "tool", "content": observation})

    # One repair retry (docs/design.md §5.3): feed the error back, ask again.
    # The retry is implemented by appending guidance and letting the next loop
    # iteration regenerate; if it fails twice in a row we stop.
    def _handle_invalid(self, messages: List[Message], raw: str, error: str):
        already_retried = any(
            m["role"] == "tool" and m["content"].startswith("[FORMAT ERROR]") for m in messages
        )
        messages.append({"role": "assistant", "content": raw})
        messages.append(
            {
                "role": "tool",
                "content": (
                    f"[FORMAT ERROR] {error}. Respond with a single valid JSON object "
                    f'containing either "action" (tool + arguments) or "final_answer".'
                ),
            }
        )
        step = Step(kind="error", observation=error, raw=raw)
        return step, already_retried
