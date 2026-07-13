"""Model backend abstraction.

The agent loop only needs a way to turn a chat transcript into the model's
next text turn. Everything else (MLX, a mock, a dev bridge to the Mac) lives
behind this interface. Mirrors the Swift `ModelBackend` protocol so the app
can swap a stub backend into the iOS Simulator (docs/design.md §3.4).
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Dict, List

# A message is {"role": "system"|"user"|"assistant"|"tool", "content": str}
Message = Dict[str, str]


@dataclass
class GenerationConfig:
    max_tokens: int = 512
    temperature: float = 0.0  # deterministic tool-calling by default


class ModelBackend(abc.ABC):
    """Turns a message transcript into the next assistant text turn."""

    name: str = "abstract"

    @abc.abstractmethod
    def generate(self, messages: List[Message], config: GenerationConfig) -> str:
        ...
