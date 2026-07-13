"""A rule-based reference policy used as a stand-in "competent model".

It emits well-formed decisions so the eval harness produces real pass/fail and
validity numbers on Linux/CI where MLX cannot run. It is a TEST FIXTURE, not
part of the product: on a Mac you swap in MLXBackend and run the same tasks to
get the real model's numbers.
"""
from __future__ import annotations

import json
import re
from typing import List

from agentcore.backends.base import Message


def _first_user(messages: List[Message]) -> str:
    for m in messages:
        if m["role"] == "user":
            return m["content"]
    return ""


def _final(answer: str) -> str:
    return json.dumps({"thought": "task done", "final_answer": answer})


def _action(tool: str, arguments: dict) -> str:
    return json.dumps({"thought": f"use {tool}", "action": {"tool": tool, "arguments": arguments}})


def reference_policy(messages: List[Message]) -> str:
    last = messages[-1]
    # After a successful observation, summarize and finish.
    if last["role"] == "tool" and last["content"].startswith("[OK]"):
        return _final(last["content"][len("[OK] ") :].strip() or "Done.")
    if last["role"] == "tool" and last["content"].startswith("[ERROR]"):
        return _final("Sorry, I couldn't complete that: " + last["content"])

    req = _first_user(messages).lower()

    # Query intent must be checked before create intent ("reminders" contains "remind").
    if "reminder" in req and ("what" in req or "list" in req or "do i have" in req):
        return _action("list_reminders", {})

    if "remind" in req:
        time_match = re.search(r"at ([0-9]{1,2}\s*(?:am|pm)?)", req)
        time = time_match.group(1).replace(" ", "") if time_match else ""
        text = re.sub(r"^.*?remind me to ", "", req)
        text = re.sub(r"\s*at [0-9]{1,2}\s*(?:am|pm)?\.?$", "", text).strip(" .")
        return _action("create_reminder", {"text": text or "reminder", "time": time})

    if "timer" in req:
        num = re.search(r"(\d+)", req)
        minutes = int(num.group(1)) if num else 5
        return _action("start_timer", {"minutes": minutes})

    if req.startswith("text ") or "send a message" in req or "message" in req:
        name = re.search(r"text (\w+)", req)
        to = name.group(1).capitalize() if name else "Unknown"
        body = re.sub(r"^.*?that ", "", req).strip(" .") or "(no message)"
        return _action("send_message", {"to": to, "body": body})

    if "note" in req or "wifi" in req or "find" in req or "search" in req:
        q = "wifi password" if "wifi" in req else re.sub(r"^.*?(find|search)\s+", "", req).strip(" .")
        return _action("search_notes", {"query": q or "note"})

    return _final("I'm not sure how to help with that using my available tools.")
