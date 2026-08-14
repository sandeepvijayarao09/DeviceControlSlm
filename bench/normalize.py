"""Normalization rules applied before argument matching and state checks.

Making "exact match" a well-defined, tested operation (eval-protocol.md §4) is
what lets us report Argument Exact-Match and state-based Task Success as real,
reproducible numbers instead of brittle string comparisons.
"""
from __future__ import annotations

import re

_WORD_NUM = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "quarter": 15, "half": 30,
}


def norm_text(s: str) -> str:
    """Case-fold, collapse whitespace, strip surrounding punctuation."""
    return re.sub(r"\s+", " ", str(s).strip().lower()).strip(" .!,")


def norm_entity(s: str) -> str:
    return norm_text(s)


def norm_bool(v) -> bool | None:
    """Accept true/false, on/off, yes/no, enable/disable, 1/0."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    t = norm_text(v)
    if t in ("true", "on", "yes", "enable", "enabled", "1"):
        return True
    if t in ("false", "off", "no", "disable", "disabled", "0"):
        return False
    return None


def norm_duration_minutes(s) -> int | None:
    """'10', '10 minutes', 'ten min', 'an hour', '1.5 hours' -> integer minutes."""
    if isinstance(s, (int, float)):
        return int(s)
    t = norm_text(s)
    if not t:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)\s*(hour|hr|h|minute|min|m|second|sec|s)?", t)
    if m:
        val = float(m.group(1))
        unit = m.group(2) or "minute"
    else:
        w = next((w for w in t.split() if w in _WORD_NUM), None)
        if w is None:
            return None
        val = _WORD_NUM[w]
        unit = "hour" if "hour" in t else ("second" if "second" in t else "minute")
    if unit.startswith(("hour", "hr", "h")):
        return int(round(val * 60))
    if unit.startswith(("second", "sec", "s")):
        return max(1, int(round(val / 60)))
    return int(round(val))


def norm_time_24h(s) -> str | None:
    """Parse a clock time to 'HH:MM'. Handles '6pm', '18:00', '6:30 pm',
    'quarter past 9', 'half past 8', 'noon', 'midnight'."""
    if s is None:
        return None
    t = norm_text(s)
    if not t:
        return None
    if "noon" in t:
        return "12:00"
    if "midnight" in t:
        return "00:00"

    ampm = None
    if re.search(r"pm\b|p\.m\.?", t):
        ampm = "pm"
    elif re.search(r"am\b|a\.m\.?", t):
        ampm = "am"
    # strip the am/pm marker so the hour digit isn't glued to letters ("6pm")
    t = re.sub(r"\s*(a\.?m\.?|p\.?m\.?)\b", " ", t).strip()

    # "quarter past 9", "half past 8", "quarter to 10"
    m = re.search(r"(quarter|half|\d+)\s+(past|to)\s+(\d{1,2})", t)
    if m:
        amt = _WORD_NUM.get(m.group(1), None)
        if amt is None and m.group(1).isdigit():
            amt = int(m.group(1))
        hour = int(m.group(3))
        minute = amt or 0
        if m.group(2) == "to":
            hour = (hour - 1) % 24
            minute = 60 - minute
        return _apply_ampm(hour, minute, ampm)

    m = re.search(r"(\d{1,2}):(\d{2})", t)
    if m:
        return _apply_ampm(int(m.group(1)), int(m.group(2)), ampm)
    m = re.search(r"(\d{1,2})", t)
    if m:
        return _apply_ampm(int(m.group(1)), 0, ampm)
    return None


def _apply_ampm(hour: int, minute: int, ampm: str | None) -> str:
    if ampm == "pm" and hour < 12:
        hour += 12
    elif ampm == "am" and hour == 12:
        hour = 0
    hour %= 24
    minute %= 60
    return f"{hour:02d}:{minute:02d}"
