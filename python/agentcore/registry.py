"""Tool registry: holds the tools in scope for a task and validates calls.

Validation is intentionally dependency-free (a small JSON-Schema subset:
type, required, properties, enum). That keeps the reference implementation
portable and mirrors what a lightweight on-device validator would enforce.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from .tool import Tool, ToolResult


class ToolRegistry:
    def __init__(self, tools: Optional[List[Tool]] = None) -> None:
        self._tools: Dict[str, Tool] = {}
        for t in tools or []:
            self.add(t)

    def add(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def names(self) -> List[str]:
        return list(self._tools)

    def specs(self) -> List[Dict[str, Any]]:
        return [t.spec() for t in self._tools.values()]

    def render_for_prompt(self) -> str:
        """Compact, model-friendly listing of the tools and their schemas."""
        lines: List[str] = []
        for t in self._tools.values():
            lines.append(f"- {t.name}: {t.description}")
            lines.append(f"  schema: {json.dumps(t.parameters, separators=(',', ':'))}")
        return "\n".join(lines)

    # --- validation -------------------------------------------------------
    def validate(self, name: str, arguments: Dict[str, Any]) -> Tuple[bool, str]:
        tool = self.get(name)
        if tool is None:
            return False, f"unknown tool '{name}'. valid tools: {', '.join(self.names())}"
        if not isinstance(arguments, dict):
            return False, "arguments must be a JSON object"
        return _validate_schema(arguments, tool.parameters)


def _validate_schema(value: Any, schema: Dict[str, Any], path: str = "arguments") -> Tuple[bool, str]:
    expected = schema.get("type")
    if expected == "object" or "properties" in schema:
        if not isinstance(value, dict):
            return False, f"{path}: expected object"
        props: Dict[str, Any] = schema.get("properties", {})
        for req in schema.get("required", []):
            if req not in value:
                return False, f"{path}: missing required field '{req}'"
        for key, sub in value.items():
            if key in props:
                ok, err = _validate_schema(sub, props[key], f"{path}.{key}")
                if not ok:
                    return ok, err
        return True, ""
    if expected == "string":
        if not isinstance(value, str):
            return False, f"{path}: expected string"
    elif expected == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            return False, f"{path}: expected integer"
    elif expected == "number":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return False, f"{path}: expected number"
    elif expected == "boolean":
        if not isinstance(value, bool):
            return False, f"{path}: expected boolean"
    elif expected == "array":
        if not isinstance(value, list):
            return False, f"{path}: expected array"
    if "enum" in schema and value not in schema["enum"]:
        return False, f"{path}: must be one of {schema['enum']}"
    return True, ""
