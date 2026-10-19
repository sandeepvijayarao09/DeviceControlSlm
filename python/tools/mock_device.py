"""Mock device tools backed by in-memory state.

These stand in for the real platform adapters (EventKit reminders, timers,
Messages, on-device search) so the agent loop can be exercised and evaluated
without a device. Each mirrors the shape of a real Tool: name, description,
JSON-Schema parameters, sensitivity, and an invoke function. On Apple, the
same names/schemas map to App Intents / framework calls (docs/design.md §5.4).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from agentcore.tool import Sensitivity, Tool, ToolResult


@dataclass
class DeviceState:
    """In-memory stand-in for on-device state; inspectable by tests/eval."""

    reminders: List[Dict[str, str]] = field(default_factory=list)
    timers: List[Dict[str, Any]] = field(default_factory=list)
    sent_messages: List[Dict[str, str]] = field(default_factory=list)
    notes: Dict[str, str] = field(default_factory=dict)


def build_tools(state: DeviceState) -> List[Tool]:
    def create_reminder(args: Dict[str, Any]) -> ToolResult:
        entry = {"text": args["text"], "time": args.get("time", "")}
        state.reminders.append(entry)
        when = f" at {entry['time']}" if entry["time"] else ""
        return ToolResult(ok=True, content=f"reminder created: '{entry['text']}'{when}", data=entry)

    def list_reminders(args: Dict[str, Any]) -> ToolResult:
        if not state.reminders:
            return ToolResult(ok=True, content="no reminders")
        listing = "; ".join(f"{r['text']} ({r['time'] or 'no time'})" for r in state.reminders)
        return ToolResult(ok=True, content=listing, data={"reminders": state.reminders})

    def start_timer(args: Dict[str, Any]) -> ToolResult:
        entry = {"minutes": args["minutes"], "label": args.get("label", "")}
        state.timers.append(entry)
        return ToolResult(ok=True, content=f"timer started for {entry['minutes']} min", data=entry)

    def send_message(args: Dict[str, Any]) -> ToolResult:
        entry = {"to": args["to"], "body": args["body"]}
        state.sent_messages.append(entry)
        return ToolResult(ok=True, content=f"message sent to {entry['to']}", data=entry)

    def search_notes(args: Dict[str, Any]) -> ToolResult:
        q = args["query"].lower()
        hits = {k: v for k, v in state.notes.items() if q in k.lower() or q in v.lower()}
        if not hits:
            return ToolResult(ok=True, content=f"no notes match '{args['query']}'")
        listing = "; ".join(f"{k}: {v}" for k, v in hits.items())
        return ToolResult(ok=True, content=listing, data={"matches": hits})

    return [
        Tool(
            name="create_reminder",
            description="Create a reminder with text and an optional time.",
            parameters={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "time": {"type": "string"},
                },
                "required": ["text"],
            },
            invoke=create_reminder,
            sensitivity=Sensitivity.CONFIRM,
        ),
        Tool(
            name="list_reminders",
            description="List all existing reminders.",
            parameters={"type": "object", "properties": {}},
            invoke=list_reminders,
            sensitivity=Sensitivity.SAFE,
        ),
        Tool(
            name="start_timer",
            description="Start a countdown timer for a number of minutes.",
            parameters={
                "type": "object",
                "properties": {
                    "minutes": {"type": "integer"},
                    "label": {"type": "string"},
                },
                "required": ["minutes"],
            },
            invoke=start_timer,
            sensitivity=Sensitivity.CONFIRM,
        ),
        Tool(
            name="send_message",
            description="Send a text message to a contact.",
            parameters={
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["to", "body"],
            },
            invoke=send_message,
            sensitivity=Sensitivity.CONFIRM,
        ),
        Tool(
            name="search_notes",
            description="Search the user's on-device notes by keyword.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            invoke=search_notes,
            sensitivity=Sensitivity.SAFE,
        ),
    ]
