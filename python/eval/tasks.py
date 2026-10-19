"""Evaluation tasks.

Each task is a natural-language request plus a `check` that inspects the final
DeviceState and AgentResult to decide success. Tasks are backend-agnostic: run
them against MockBackend (here/CI) or MLXBackend (on a Mac) with the same
criteria, which is how Python eval gains transfer to the shipped app
(docs/design.md §8, ADR-0006).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from agentcore import AgentResult
from tools import DeviceState

Check = Callable[[AgentResult, DeviceState], bool]


@dataclass
class Task:
    id: str
    request: str
    check: Check


TASKS = [
    Task(
        id="reminder_with_time",
        request="Remind me to call mom at 6pm.",
        check=lambda r, s: any(
            "mom" in x["text"].lower() and "6" in x["time"] for x in s.reminders
        ),
    ),
    Task(
        id="start_timer",
        request="Set a timer for 10 minutes.",
        check=lambda r, s: any(t["minutes"] == 10 for t in s.timers),
    ),
    Task(
        id="list_reminders_readonly",
        request="What reminders do I have?",
        check=lambda r, s: any(
            st.tool == "list_reminders" for st in r.tool_calls
        )
        and r.answer is not None,
    ),
    Task(
        id="send_message",
        request="Text Alex that I'll be 10 minutes late.",
        check=lambda r, s: any(
            m["to"].lower() == "alex" and "late" in m["body"].lower()
            for m in s.sent_messages
        ),
    ),
    Task(
        id="search_notes",
        request="Find the wifi password in my notes.",
        check=lambda r, s: any(
            st.tool == "search_notes" for st in r.tool_calls
        )
        and r.answer is not None,
    ),
]
