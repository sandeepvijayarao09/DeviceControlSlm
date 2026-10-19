"""Deterministic mock backend for running the agent loop without a GPU.

Two modes:
  * scripted  - a fixed list of turns, returned in order (great for tests/eval).
  * policy    - a callable(messages) -> str, for rule-based simulation.

This is what makes the agent logic verifiable in CI / on Linux / in the iOS
Simulator, where real MLX inference cannot run (docs/design.md §3.4).
"""
from __future__ import annotations

from typing import Callable, List, Optional

from .base import GenerationConfig, Message, ModelBackend

Policy = Callable[[List[Message]], str]


class MockBackend(ModelBackend):
    name = "mock"

    def __init__(
        self,
        scripted: Optional[List[str]] = None,
        policy: Optional[Policy] = None,
    ) -> None:
        if (scripted is None) == (policy is None):
            raise ValueError("provide exactly one of `scripted` or `policy`")
        self._scripted = list(scripted) if scripted is not None else None
        self._policy = policy
        self._i = 0

    def generate(self, messages: List[Message], config: GenerationConfig) -> str:
        if self._policy is not None:
            return self._policy(messages)
        assert self._scripted is not None
        if self._i >= len(self._scripted):
            # Nothing left to say: end the task cleanly rather than loop.
            return '{"thought": "done", "final_answer": "(mock ran out of scripted turns)"}'
        turn = self._scripted[self._i]
        self._i += 1
        return turn
