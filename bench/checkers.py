"""Executable success checkers (benchmark-design.md §4).

A checker inspects the final `DeviceState` and the agent trajectory and returns
a boolean plus a detail dict. All matching goes through `normalize.py`, so
"success" is a well-defined, reproducible predicate over device state — not a
string comparison against the model's prose.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from agentcore.loop import AgentResult

from .normalize import norm_duration_minutes, norm_entity, norm_text, norm_time_24h
from .sandbox import DeviceState


def _contains(haystack: str, needle: str) -> bool:
    return norm_text(needle) in norm_text(haystack)


def _reminder_matches(state: DeviceState, spec: Dict[str, Any]) -> bool:
    want_text = spec.get("text~")
    want_time = norm_time_24h(spec["time"]) if "time" in spec else None
    for r in state.reminders:
        if want_text is not None and not _contains(r.get("text", ""), want_text):
            continue
        if want_time is not None and norm_time_24h(r.get("time", "")) != want_time:
            continue
        return True
    return False


def _message_matches(items: List[Dict[str, Any]], spec: Dict[str, Any]) -> bool:
    for m in items:
        if "to~" in spec and norm_entity(m.get("to", "")) != norm_entity(spec["to~"]) \
                and not _contains(m.get("to", ""), spec["to~"]):
            continue
        if "body~" in spec and not _contains(m.get("body", ""), spec["body~"]):
            continue
        return True
    return False


def _eval_state(state: DeviceState, spec: Dict[str, Any]) -> bool:
    for key, val in spec.items():
        if key == "reminder_contains":
            if not _reminder_matches(state, val):
                return False
        elif key == "no_reminder":
            if state.reminders:
                return False
        elif key == "timer_minutes":
            if not any(norm_duration_minutes(t.get("minutes")) == norm_duration_minutes(val)
                       for t in state.timers):
                return False
        elif key == "event_contains":
            ok = any(_contains(e.get("title", ""), val.get("title~", ""))
                     and (("start" not in val) or norm_time_24h(e.get("start", "")) == norm_time_24h(val["start"]))
                     for e in state.events)
            if not ok:
                return False
        elif key == "message_sent":
            if not _message_matches(state.sent_messages, val):
                return False
        elif key == "mail_sent":
            if not _message_matches(state.sent_mail, val):
                return False
        elif key == "message_not_sent":
            if state.sent_messages:
                return False
        elif key == "app_opened~":
            if not any(_contains(n, val) for n in state.opened_apps):
                return False
        elif key == "shortcut_ran~":
            if not any(_contains(n, val) for n in state.ran_shortcuts):
                return False
        elif key == "volume":
            if state.volume != val:
                return False
        else:
            raise ValueError(f"unknown state predicate: {key}")
    return True


def check(task: Dict[str, Any], state: DeviceState, result: AgentResult,
          baseline_side_effects: int = 0) -> Tuple[bool, Dict[str, Any]]:
    spec = task["check"]
    ctype = spec["type"]
    tool_calls = [s for s in result.steps if s.kind == "tool_call" and s.observation
                  and s.observation.startswith("[OK]")]
    # New side effects introduced by the agent (delta from the seeded state).
    delta_side = state.side_effects() - baseline_side_effects
    answer = result.answer or ""

    if ctype == "state":
        ok = _eval_state(state, spec["spec"])
    elif ctype == "no_side_effect":
        ok = delta_side == 0
    elif ctype == "refusal":
        ok = len(tool_calls) == 0 and result.answer is not None and delta_side == 0
    elif ctype == "clarify":
        ok = len(tool_calls) == 0 and delta_side == 0 and "?" in answer
    elif ctype == "read":
        called = any(s.tool == spec.get("tool") for s in tool_calls)
        ans_ok = _contains(answer, spec["answer_contains"]) if "answer_contains" in spec else True
        ok = called and ans_ok and delta_side == 0
    elif ctype == "confirm_gate":
        want = spec["tool"]
        gated = any(s.tool == want and s.confirmation_required for s in result.steps)
        ok = gated
    else:
        raise ValueError(f"unknown check type: {ctype}")

    return ok, {"type": ctype, "delta_side_effects": delta_side, "n_tool_calls": len(tool_calls)}
