"""Parse the model's single-JSON-object turn into a structured decision.

Small models often wrap JSON in prose or code fences despite instructions, so
we extract the first balanced JSON object rather than trusting `json.loads`
on the whole string. This is the software-side complement to constrained
decoding (docs/design.md §5.3): even without a constrained sampler, we recover
a usable object when one is present.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class Decision:
    thought: str = ""
    tool: Optional[str] = None
    arguments: Optional[Dict[str, Any]] = None
    final_answer: Optional[str] = None
    parse_error: Optional[str] = None

    @property
    def is_tool_call(self) -> bool:
        return self.tool is not None

    @property
    def is_final(self) -> bool:
        return self.final_answer is not None


def _extract_first_json_object(text: str) -> Optional[str]:
    """Return the substring of the first balanced top-level {...} object."""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start : i + 1]
        start = text.find("{", start + 1)
    return None


def parse_decision(text: str) -> Decision:
    raw = _extract_first_json_object(text)
    if raw is None:
        return Decision(parse_error="no JSON object found in model output")
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        return Decision(parse_error=f"invalid JSON: {e}")
    if not isinstance(obj, dict):
        return Decision(parse_error="top-level JSON must be an object")

    thought = obj.get("thought", "")
    thought = thought if isinstance(thought, str) else ""

    if "final_answer" in obj:
        fa = obj["final_answer"]
        return Decision(thought=thought, final_answer=str(fa))

    action = obj.get("action")
    if isinstance(action, dict) and "tool" in action:
        args = action.get("arguments", {})
        if not isinstance(args, dict):
            return Decision(thought=thought, parse_error="action.arguments must be an object")
        return Decision(thought=thought, tool=str(action["tool"]), arguments=args)

    # Some models flatten `tool`/`arguments` to the top level. Accept that too.
    if "tool" in obj:
        args = obj.get("arguments", {})
        if not isinstance(args, dict):
            return Decision(thought=thought, parse_error="arguments must be an object")
        return Decision(thought=thought, tool=str(obj["tool"]), arguments=args)

    return Decision(thought=thought, parse_error="object has neither 'action' nor 'final_answer'")
