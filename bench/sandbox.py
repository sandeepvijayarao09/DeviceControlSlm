"""Executable device sandbox: builds runtime Tools from the canonical catalog
and applies their side effects to inspectable in-memory state.

The same catalog + sensitivity that drive this sandbox drive the shipped app's
tool registry, so a task that passes here exercises the real agent contract.
Success is judged by the resulting `DeviceState` (benchmark-design.md §4), not
by string-matching the model's text.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from agentcore.tool import Sensitivity, Tool, ToolResult

CATALOG_PATH = Path(__file__).resolve().parent / "tools" / "catalog.json"


def load_catalog() -> Dict[str, Any]:
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


@dataclass
class DeviceState:
    reminders: List[Dict[str, Any]] = field(default_factory=list)
    timers: List[Dict[str, Any]] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)
    sent_messages: List[Dict[str, Any]] = field(default_factory=list)
    sent_mail: List[Dict[str, Any]] = field(default_factory=list)
    notes: Dict[str, str] = field(default_factory=dict)
    opened_apps: List[str] = field(default_factory=list)
    ran_shortcuts: List[str] = field(default_factory=list)
    volume: int | None = None

    def snapshot(self) -> Dict[str, Any]:
        return copy.deepcopy(self.__dict__)

    def side_effects(self) -> int:
        """Count of mutations that represent real actions (for no_side_effect checks)."""
        return (
            len(self.reminders) + len(self.timers) + len(self.events)
            + len(self.sent_messages) + len(self.sent_mail)
            + len(self.opened_apps) + len(self.ran_shortcuts)
            + (0 if self.volume is None else 1)
        )


_SENS = {"safe": Sensitivity.SAFE, "confirm": Sensitivity.CONFIRM, "blocked": Sensitivity.BLOCKED}


def _implementations(state: DeviceState):
    def create_reminder(a):
        e = {"text": a["text"], "time": a.get("time", "")}
        state.reminders.append(e)
        return ToolResult(True, f"reminder created: {e['text']}", e)

    def list_reminders(a):
        if not state.reminders:
            return ToolResult(True, "no reminders")
        return ToolResult(True, "; ".join(f"{r['text']} ({r['time'] or 'no time'})" for r in state.reminders),
                          {"reminders": state.reminders})

    def delete_reminder(a):
        before = len(state.reminders)
        key = a["text"].lower()
        state.reminders = [r for r in state.reminders if key not in r["text"].lower()]
        return ToolResult(True, f"deleted {before - len(state.reminders)} reminder(s)")

    def start_timer(a):
        e = {"minutes": a["minutes"], "label": a.get("label", "")}
        state.timers.append(e)
        return ToolResult(True, f"timer started for {e['minutes']} min", e)

    def create_calendar_event(a):
        e = {"title": a["title"], "start": a["start"], "end": a.get("end", "")}
        state.events.append(e)
        return ToolResult(True, f"event '{e['title']}' at {e['start']}", e)

    def list_calendar_events(a):
        if not state.events:
            return ToolResult(True, "no events")
        return ToolResult(True, "; ".join(f"{e['title']} at {e['start']}" for e in state.events),
                          {"events": state.events})

    def send_message(a):
        e = {"to": a["to"], "body": a["body"]}
        state.sent_messages.append(e)
        return ToolResult(True, f"message sent to {e['to']}", e)

    def compose_mail(a):
        e = {"to": a["to"], "subject": a.get("subject", ""), "body": a["body"]}
        state.sent_mail.append(e)
        return ToolResult(True, f"email sent to {e['to']}", e)

    def search_notes(a):
        q = a["query"].lower()
        hits = {k: v for k, v in state.notes.items() if q in k.lower() or q in v.lower()}
        if not hits:
            return ToolResult(True, f"no notes match '{a['query']}'")
        return ToolResult(True, "; ".join(f"{k}: {v}" for k, v in hits.items()), {"matches": hits})

    def open_app(a):
        state.opened_apps.append(a["name"])
        return ToolResult(True, f"opened {a['name']}")

    def run_shortcut(a):
        state.ran_shortcuts.append(a["name"])
        return ToolResult(True, f"ran shortcut {a['name']}")

    def set_volume(a):
        state.volume = a["level"]
        return ToolResult(True, f"volume set to {a['level']}%")

    return {
        "create_reminder": create_reminder, "list_reminders": list_reminders,
        "delete_reminder": delete_reminder, "start_timer": start_timer,
        "create_calendar_event": create_calendar_event, "list_calendar_events": list_calendar_events,
        "send_message": send_message, "compose_mail": compose_mail,
        "search_notes": search_notes, "open_app": open_app,
        "run_shortcut": run_shortcut, "set_volume": set_volume,
    }


def build_tools(state: DeviceState, subset: List[str] | None = None) -> List[Tool]:
    """Instantiate runtime Tools from the catalog, limited to `subset` if given."""
    catalog = load_catalog()
    impls = _implementations(state)
    tools: List[Tool] = []
    for spec in catalog["tools"]:
        name = spec["name"]
        if subset is not None and name not in subset:
            continue
        impl = impls.get(name)
        if impl is None:
            continue
        tools.append(Tool(
            name=name,
            description=spec["description"],
            parameters=spec["parameters"],
            invoke=impl,
            sensitivity=_SENS[spec["sensitivity"]],
        ))
    return tools


def seed_state(initial: Dict[str, Any] | None) -> DeviceState:
    st = DeviceState()
    for k, v in (initial or {}).items():
        if hasattr(st, k):
            setattr(st, k, copy.deepcopy(v))
    return st
