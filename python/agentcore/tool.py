"""Tool abstraction: the contract between the agent and any capability.

Mirrors the Swift `Tool` protocol described in docs/design.md §5.1 so the
tool schemas and semantics transfer directly to the on-device app.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional


class Sensitivity(enum.Enum):
    """How much scrutiny a tool call needs before it runs.

    SAFE:    no side effects worth confirming (queries, reads).
    CONFIRM: has real/irreversible side effects -> requires user confirmation
             in the UI before execution (docs/design.md ADR-0005).
    BLOCKED: never allowed; compiled out / hard-refused.
    """

    SAFE = "safe"
    CONFIRM = "confirm"
    BLOCKED = "blocked"


@dataclass
class ToolResult:
    """The observation returned to the agent after a tool runs."""

    ok: bool
    content: str
    data: Dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        status = "OK" if self.ok else "ERROR"
        return f"[{status}] {self.content}"


@dataclass
class Tool:
    """A single capability exposed to the model.

    `parameters` is a JSON Schema (draft-07 style) describing `arguments`.
    `invoke` receives already-validated arguments and returns a ToolResult.
    """

    name: str
    description: str
    parameters: Dict[str, Any]
    invoke: Callable[[Dict[str, Any]], ToolResult]
    sensitivity: Sensitivity = Sensitivity.SAFE

    def spec(self) -> Dict[str, Any]:
        """Machine-readable spec advertised to the model."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }
