"""AgentCore: portable on-device agent loop, tool system, and model backends.

Reference implementation in Python for fast iteration and evaluation; the same
design ports to Swift for the shipping app (docs/design.md, ADR-0006).
"""
from .loop import Agent, AgentResult, Step
from .parser import Decision, parse_decision
from .registry import ToolRegistry
from .tool import Sensitivity, Tool, ToolResult

__all__ = [
    "Agent",
    "AgentResult",
    "Step",
    "Decision",
    "parse_decision",
    "ToolRegistry",
    "Tool",
    "ToolResult",
    "Sensitivity",
]
