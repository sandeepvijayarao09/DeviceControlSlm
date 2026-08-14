"""Executable device sandbox: builds runtime Tools from the canonical catalog
and applies their side effects to inspectable state.

The device is modelled as (a) a dict of typed *settings* (wifi, brightness,
volume, focus, ...) and (b) *collections/logs* of discrete actions (alarms,
timers, reminders, messages, opened apps, media actions, ...). Success is judged
by this state (benchmark-design.md §4), not by the model's prose. The device is
idealized/permissive: every command is executable in simulation; per-OS
permission gating is a later concern.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from agentcore.tool import Sensitivity, Tool, ToolResult

from .normalize import norm_bool

CATALOG_PATH = Path(__file__).resolve().parent / "tools" / "catalog.json"


def load_catalog() -> Dict[str, Any]:
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def _default_settings() -> Dict[str, Any]:
    return {
        "wifi": True, "bluetooth": True, "airplane_mode": False, "cellular_data": True,
        "hotspot": False, "do_not_disturb": False, "focus_mode": "off",
        "brightness": 70, "dark_mode": False, "auto_lock": 5, "orientation_lock": False,
        "wallpaper": "default", "volume": 50, "mute": False, "ringer_mode": "ring",
        "low_power_mode": False, "location_services": True, "flashlight": False,
        # read-only telemetry
        "battery": 82, "storage_free_gb": 64, "powered": True,
    }


@dataclass
class DeviceState:
    settings: Dict[str, Any] = field(default_factory=_default_settings)
    alarms: List[Dict[str, Any]] = field(default_factory=list)
    timers: List[Dict[str, Any]] = field(default_factory=list)
    reminders: List[Dict[str, Any]] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)
    sent_messages: List[Dict[str, Any]] = field(default_factory=list)
    sent_mail: List[Dict[str, Any]] = field(default_factory=list)
    calls: List[Dict[str, Any]] = field(default_factory=list)
    opened_apps: List[str] = field(default_factory=list)
    closed_apps: List[str] = field(default_factory=list)
    media_actions: List[str] = field(default_factory=list)
    screenshots: int = 0
    clipboard: str = ""
    notes: Dict[str, str] = field(default_factory=dict)

    def snapshot(self) -> Dict[str, Any]:
        return copy.deepcopy(self.__dict__)


_SENS = {"safe": Sensitivity.SAFE, "confirm": Sensitivity.CONFIRM, "blocked": Sensitivity.BLOCKED}


def _implementations(state: DeviceState):
    def _set(key):
        def f(a):
            state.settings[key] = norm_bool(a["enabled"])
            return ToolResult(True, f"{key} set to {state.settings[key]}", {key: state.settings[key]})
        return f

    def set_focus_mode(a):
        state.settings["focus_mode"] = a["mode"]
        return ToolResult(True, f"focus mode: {a['mode']}")

    def set_brightness(a):
        state.settings["brightness"] = int(a["level"])
        return ToolResult(True, f"brightness {a['level']}%")

    def set_auto_lock(a):
        state.settings["auto_lock"] = int(a["minutes"])
        return ToolResult(True, f"auto-lock {a['minutes']} min")

    def set_wallpaper(a):
        state.settings["wallpaper"] = a["name"]
        return ToolResult(True, f"wallpaper {a['name']}")

    def take_screenshot(a):
        state.screenshots += 1
        return ToolResult(True, "screenshot captured")

    def set_volume(a):
        state.settings["volume"] = int(a["level"])
        return ToolResult(True, f"volume {a['level']}%")

    def set_ringer_mode(a):
        state.settings["ringer_mode"] = a["mode"]
        return ToolResult(True, f"ringer {a['mode']}")

    def restart_device(a):
        return ToolResult(True, "device restarting")

    def shutdown_device(a):
        state.settings["powered"] = False
        return ToolResult(True, "device powering off")

    def set_alarm(a):
        e = {"time": a["time"], "label": a.get("label", "")}
        state.alarms.append(e)
        return ToolResult(True, f"alarm set for {a['time']}", e)

    def start_timer(a):
        e = {"minutes": int(a["minutes"]), "label": a.get("label", "")}
        state.timers.append(e)
        return ToolResult(True, f"timer {a['minutes']} min", e)

    def cancel_timers(a):
        n = len(state.timers)
        state.timers.clear()
        return ToolResult(True, f"cancelled {n} timer(s)")

    def create_reminder(a):
        e = {"text": a["text"], "time": a.get("time", "")}
        state.reminders.append(e)
        return ToolResult(True, f"reminder: {a['text']}", e)

    def list_reminders(a):
        if not state.reminders:
            return ToolResult(True, "no reminders")
        return ToolResult(True, "; ".join(f"{r['text']} ({r['time'] or 'no time'})" for r in state.reminders),
                          {"reminders": state.reminders})

    def create_calendar_event(a):
        e = {"title": a["title"], "start": a["start"], "end": a.get("end", "")}
        state.events.append(e)
        return ToolResult(True, f"event '{a['title']}' at {a['start']}", e)

    def list_calendar_events(a):
        if not state.events:
            return ToolResult(True, "no events")
        return ToolResult(True, "; ".join(f"{e['title']} at {e['start']}" for e in state.events),
                          {"events": state.events})

    def send_message(a):
        e = {"to": a["to"], "body": a["body"]}
        state.sent_messages.append(e)
        return ToolResult(True, f"message sent to {a['to']}", e)

    def compose_mail(a):
        e = {"to": a["to"], "subject": a.get("subject", ""), "body": a["body"]}
        state.sent_mail.append(e)
        return ToolResult(True, f"email sent to {a['to']}", e)

    def start_call(a):
        state.calls.append({"to": a["to"]})
        return ToolResult(True, f"calling {a['to']}")

    def open_app(a):
        state.opened_apps.append(a["name"])
        return ToolResult(True, f"opened {a['name']}")

    def close_app(a):
        state.closed_apps.append(a["name"])
        return ToolResult(True, f"closed {a['name']}")

    def media_control(a):
        state.media_actions.append(a["action"])
        return ToolResult(True, f"media {a['action']}")

    def play_music(a):
        q = a.get("query", "")
        state.media_actions.append("play")
        return ToolResult(True, f"playing music{' ' + q if q else ''}", {"query": q})

    def set_clipboard(a):
        state.clipboard = a["text"]
        return ToolResult(True, "copied to clipboard")

    def search_notes(a):
        q = a["query"].lower()
        hits = {k: v for k, v in state.notes.items() if q in k.lower() or q in v.lower()}
        if not hits:
            return ToolResult(True, f"no notes match '{a['query']}'")
        return ToolResult(True, "; ".join(f"{k}: {v}" for k, v in hits.items()), {"matches": hits})

    def get_battery(a):
        return ToolResult(True, f"battery at {state.settings['battery']}%", {"battery": state.settings["battery"]})

    def get_storage(a):
        return ToolResult(True, f"{state.settings['storage_free_gb']} GB free",
                          {"free_gb": state.settings["storage_free_gb"]})

    def get_connectivity_status(a):
        s = state.settings
        return ToolResult(True, f"wifi={'on' if s['wifi'] else 'off'}, "
                                f"bluetooth={'on' if s['bluetooth'] else 'off'}, "
                                f"cellular={'on' if s['cellular_data'] else 'off'}")

    return {
        "set_wifi": _set("wifi"), "set_bluetooth": _set("bluetooth"),
        "set_airplane_mode": _set("airplane_mode"), "set_cellular_data": _set("cellular_data"),
        "set_hotspot": _set("hotspot"), "set_do_not_disturb": _set("do_not_disturb"),
        "set_focus_mode": set_focus_mode, "set_brightness": set_brightness,
        "set_dark_mode": _set("dark_mode"), "set_auto_lock": set_auto_lock,
        "set_orientation_lock": _set("orientation_lock"), "set_wallpaper": set_wallpaper,
        "take_screenshot": take_screenshot, "set_volume": set_volume, "set_mute": _set("mute"),
        "set_ringer_mode": set_ringer_mode, "set_low_power_mode": _set("low_power_mode"),
        "restart_device": restart_device, "shutdown_device": shutdown_device,
        "set_alarm": set_alarm, "start_timer": start_timer, "cancel_timers": cancel_timers,
        "create_reminder": create_reminder, "list_reminders": list_reminders,
        "create_calendar_event": create_calendar_event, "list_calendar_events": list_calendar_events,
        "send_message": send_message, "compose_mail": compose_mail, "start_call": start_call,
        "open_app": open_app, "close_app": close_app, "media_control": media_control,
        "play_music": play_music, "set_flashlight": _set("flashlight"),
        "set_clipboard": set_clipboard, "set_location_services": _set("location_services"),
        "get_battery": get_battery, "get_storage": get_storage,
        "get_connectivity_status": get_connectivity_status, "search_notes": search_notes,
    }


def build_tools(state: DeviceState, subset: List[str] | None = None) -> List[Tool]:
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
            name=name, description=spec["description"], parameters=spec["parameters"],
            invoke=impl, sensitivity=_SENS[spec["sensitivity"]],
        ))
    return tools


def seed_state(initial: Dict[str, Any] | None) -> DeviceState:
    st = DeviceState()
    for k, v in (initial or {}).items():
        if k == "settings" and isinstance(v, dict):
            st.settings.update(copy.deepcopy(v))
        elif hasattr(st, k):
            setattr(st, k, copy.deepcopy(v))
    return st
