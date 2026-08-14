"""Generate supervised fine-tuning data for the device-control SLM.

Design goals (see docs/paper/research-plan.md, M5):
  * Teach the exact behaviours the base models fail at on DeviceControlBench:
    - COMPOSITIONAL multi-step requests (base collapses to 0%): every compound
      task is rolled out as a full trajectory, so the model is supervised to
      emit action 1, observe, emit action 2, ..., then finish.
    - CLARIFY on under-specification (base acts instead of asking).
    - REFUSE out-of-scope requests.
  * Use the SAME prompt/tool formatting as inference (agentcore system template
    + tool rendering; tool observations as `[tool result] ...` user turns), so
    there is zero train/inference skew.
  * NO contamination with the eval benchmark: any generated prompt colliding
    (normalized) with a DeviceControlBench prompt/paraphrase is dropped.

Output: MLX-LM chat JSONL ({"messages": [...]}) — base-agnostic; the tokenizer's
chat template is applied at train time, so one dataset trains every candidate
base.

    python -m finetune.gen_data --n 2400 --out finetune/data
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from agentcore import ToolRegistry
from agentcore.loop import _PROMPT_PATH
from bench.dataset import load_tasks
from bench.normalize import norm_time_24h
from bench.sandbox import DeviceState, build_tools, seed_state

SYSTEM_TEMPLATE = _PROMPT_PATH.read_text(encoding="utf-8")

ALL_TOOLS = [t["name"] for t in json.loads(
    (Path(__file__).resolve().parents[1] / "bench" / "tools" / "catalog.json").read_text())["tools"]]

APPS = ["Spotify", "Safari", "Notes", "Mail", "Maps", "Calendar", "Photos",
        "Messages", "Music", "Slack", "Chrome", "Weather", "Camera", "Settings"]
NAMES = ["Alex", "Sam", "Jordan", "Taylor", "mom", "dad", "Chris", "Riley", "Casey"]
TASKS_TODO = ["buy milk", "call the plumber", "email the report", "water the plants",
              "renew the parking permit", "pick up dry cleaning", "book a table",
              "submit the form", "pay the electricity bill", "feed the cat"]


def call(tool, **args):
    return {"tool": tool, "arguments": args}


# ---- atomic action generators: each returns (phrase_fragment, gold_call) ----
def _toggle():
    opts = {
        "wifi": "set_wifi", "bluetooth": "set_bluetooth", "airplane mode": "set_airplane_mode",
        "do not disturb": "set_do_not_disturb", "low power mode": "set_low_power_mode",
        "the flashlight": "set_flashlight", "dark mode": "set_dark_mode",
        "the hotspot": "set_hotspot", "cellular data": "set_cellular_data",
        "location services": "set_location_services", "the orientation lock": "set_orientation_lock",
    }
    name, tool = random.choice(list(opts.items()))
    on = random.random() < 0.5
    verb = random.choice(["turn on", "enable", "switch on"]) if on else random.choice(
        ["turn off", "disable", "switch off"])
    return f"{verb} {name}", call(tool, enabled=on)


def _level():
    which, tool = random.choice([("brightness", "set_brightness"), ("volume", "set_volume")])
    n = random.choice([5, 10, 15, 25, 35, 45, 55, 65, 75, 85, 95, 100])
    frag = random.choice([f"set the {which} to {n}%", f"set the {which} to {n} percent",
                          f"put the {which} at {n}%"])
    return frag, call(tool, level=n)


def _alarm():
    h, m = random.randint(1, 12), random.choice([0, 15, 30, 45])
    ap = random.choice(["am", "pm"])
    surface = f"{h}:{m:02d} {ap}" if m else f"{h} {ap}"
    return f"set an alarm for {surface}", call("set_alarm", time=norm_time_24h(surface))


def _timer():
    n = random.choice([2, 3, 8, 12, 18, 22, 40, 45, 90])
    return f"set a timer for {n} minutes", call("start_timer", minutes=n)


def _app_open():
    a = random.choice(APPS)
    return f"{random.choice(['open', 'launch'])} {a}", call("open_app", name=a)


def _app_close():
    a = random.choice(APPS)
    return f"close {a}", call("close_app", name=a)


def _media():
    act, frag = random.choice([("pause", "pause the music"), ("play", "resume playback"),
                               ("next", "skip to the next song"), ("previous", "go back a track"),
                               ("stop", "stop the music")])
    return frag, call("media_control", action=act)


def _screenshot():
    return "take a screenshot", call("take_screenshot")


def _focus():
    mode = random.choice(["work", "personal", "sleep", "driving", "fitness"])
    return f"set my focus to {mode}", call("set_focus_mode", mode=mode)


def _ringer():
    mode, frag = random.choice([("silent", "put the phone on silent"),
                                ("vibrate", "switch to vibrate"), ("ring", "turn the ringer on")])
    return frag, call("set_ringer_mode", mode=mode)


def _reminder():
    t = random.choice(TASKS_TODO)
    h, m = random.randint(1, 12), random.choice([0, 30])
    ap = random.choice(["am", "pm"])
    surface = f"{h}:{m:02d} {ap}" if m else f"{h} {ap}"
    return f"remind me to {t} at {surface}", call("create_reminder", text=t, time=norm_time_24h(surface))


ACTIONS: List[Callable[[], Tuple[str, Dict[str, Any]]]] = [
    _toggle, _level, _alarm, _timer, _app_open, _app_close, _media, _screenshot,
    _focus, _ringer, _reminder,
]


def _subset_for(gold_tools: List[str], size: int = 16) -> List[str]:
    """Offer a random tool subset that always includes the gold tools + distractors."""
    needed = list(dict.fromkeys(gold_tools))
    pool = [t for t in ALL_TOOLS if t not in needed]
    random.shuffle(pool)
    subset = needed + pool[: max(0, size - len(needed))]
    random.shuffle(subset)
    return subset


def _system_message(subset: List[str], state: DeviceState) -> str:
    registry = ToolRegistry(build_tools(state, subset=subset))
    return SYSTEM_TEMPLATE.replace("{tools}", registry.render_for_prompt())


def _assistant_action(tool: str, args: Dict[str, Any]) -> str:
    return json.dumps({"thought": "taking the requested action", "action": {"tool": tool, "arguments": args}})


def _assistant_final(text: str) -> str:
    return json.dumps({"thought": "done", "final_answer": text})


def _rollout(prompt: str, gold_calls: List[Dict[str, Any]], final: str,
             initial_state: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Build a full chat trajectory, executing gold calls on a sandbox to get
    realistic observations (as user `[tool result]` turns)."""
    gold_tools = [c["tool"] for c in gold_calls]
    subset = _subset_for(gold_tools)
    state = seed_state(initial_state)
    registry = ToolRegistry(build_tools(state, subset=subset))
    messages = [{"role": "system", "content": _system_message(subset, state)},
                {"role": "user", "content": prompt}]
    for c in gold_calls:
        messages.append({"role": "assistant", "content": _assistant_action(c["tool"], c["arguments"])})
        tool = registry.get(c["tool"])
        obs = tool.invoke(c["arguments"]).render()
        messages.append({"role": "user", "content": f"[tool result]\n{obs}"})
    messages.append({"role": "assistant", "content": _assistant_final(final)})
    return {"messages": messages}


# ---- category builders -------------------------------------------------------
def gen_single():
    frag, c = random.choice(ACTIONS)()
    prompt = frag[0].upper() + frag[1:] + "."
    return _rollout(prompt, [c], "Done.")


def gen_compositional():
    k = random.choice([2, 2, 3, 3, 4])
    used_tools, frags, calls = set(), [], []
    tries = 0
    while len(calls) < k and tries < 40:
        tries += 1
        frag, c = random.choice(ACTIONS)()
        if c["tool"] in used_tools:
            continue
        used_tools.add(c["tool"])
        frags.append(frag)
        calls.append(c)
    conn = random.choice([" and ", ", then ", ", and also ", ", "])
    prompt = conn.join(frags)
    prompt = prompt[0].upper() + prompt[1:] + "."
    return _rollout(prompt, calls, "All done.")


def gen_read():
    kind = random.choice(["battery", "storage", "connectivity", "note"])
    if kind == "battery":
        return _rollout("How much battery do I have?", [call("get_battery")], "You're at 82%.")
    if kind == "storage":
        return _rollout("How much free storage is left?", [call("get_storage")], "You have 64 GB free.")
    if kind == "connectivity":
        return _rollout("Am I online?", [call("get_connectivity_status")], "Wi-Fi is on.")
    key = random.choice(["Locker", "Bike lock", "Router", "Safe"])
    val = random.choice(["code is 7391", "combo is 22-14-6", "password is bluecat42"])
    return _rollout(f"What's my {key.lower()} code?", [call("search_notes", query=key.lower())],
                    f"It's {val}.", initial_state={"notes": {key: f"{key} {val}"}})


# Parametric clarify prompts: an under-specified action (missing target/value/slot).
_CLARIFY_TEMPLATES = [
    ("Turn {it} on.", "which setting"), ("Turn {it} off.", "which setting"),
    ("Enable {it}.", "which setting"), ("Set the {sv}.", "value"),
    ("Turn the {sv} up.", "value"), ("Turn the {sv} down.", "value"),
    ("Set an alarm.", "time"), ("Wake me up.", "time"),
    ("Open {the_app}.", "which app"), ("Launch {it}.", "which app"),
    ("Close {it}.", "which app"), ("Set a timer.", "duration"),
    ("Start the countdown.", "duration"), ("Remind me.", "what_time"),
    ("Add a reminder.", "what_time"), ("Text {them}.", "who_what"),
    ("Send {them} a message.", "who_what"), ("Call {them}.", "who"),
    ("Give {them} a ring.", "who"), ("Set my focus.", "mode"),
    ("Set the ringer.", "mode"), ("Change {that}.", "which setting"),
    ("Schedule {it}.", "what_time"), ("Play {it}.", "what"),
]
_CL_IT = ["it", "that", "this", "the thing"]
_CL_SV = ["brightness", "volume"]
_CL_APP = ["the app", "it", "that one"]
_CL_THEM = ["them", "him", "her"]
_CL_QUESTIONS = {
    "which setting": ["Which setting do you mean?", "Which one should I toggle?"],
    "value": ["What level should I set it to?", "What value would you like?"],
    "time": ["What time should the alarm be?", "When should I set it for?"],
    "which app": ["Which app should I open?", "Which app do you mean?"],
    "duration": ["How long should the timer be?", "How many minutes?"],
    "what_time": ["What should I remind you about, and when?", "Remind you of what, and at what time?"],
    "who_what": ["Who should I message, and what should it say?", "Who is it for, and what's the message?"],
    "who": ["Who would you like me to call?", "Which contact should I call?"],
    "mode": ["Which mode would you like?", "Which one should I set?"],
    "what": ["What would you like me to play?", "Anything specific to play?"],
}


def gen_clarify():
    tmpl, slot = random.choice(_CLARIFY_TEMPLATES)
    prompt = tmpl.format(it=random.choice(_CL_IT), sv=random.choice(_CL_SV),
                         the_app=random.choice(_CL_APP), them=random.choice(_CL_THEM),
                         that=random.choice(_CL_IT))
    prompt = prompt[0].upper() + prompt[1:]
    reply = random.choice(_CL_QUESTIONS[slot])
    return {"messages": [
        {"role": "system", "content": _system_message(_subset_for([]), DeviceState())},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": _assistant_final(reply)},
    ]}


# Parametric refusals: out-of-scope requests with varied objects.
_PLATFORMS = ["Instagram", "TikTok", "Twitter", "Facebook", "Snapchat", "LinkedIn"]
_FOODS = ["a pizza", "sushi", "tacos", "a burger", "groceries"]
_CITIES = ["Rome", "Tokyo", "Paris", "Denver", "Lagos", "Oslo"]
_TICKERS = ["Apple", "Tesla", "Nvidia", "Amazon"]


def _refuse_prompt():
    kind = random.randint(0, 10)
    amt = random.choice(["$50", "$200", "$1000", "$5000", "500 dollars"])
    if kind == 0:
        return f"Transfer {amt} to {random.choice(NAMES)}."
    if kind == 1:
        return f"Wire {amt} to this account."
    if kind == 2:
        return f"Post {random.choice(['this', 'it', 'my photo'])} to {random.choice(_PLATFORMS)}."
    if kind == 3:
        return f"Order {random.choice(_FOODS)} for {random.choice(['tonight', 'lunch', 'dinner'])}."
    if kind == 4:
        return f"Book a {random.choice(['flight', 'hotel', 'rental car'])} to {random.choice(_CITIES)}."
    if kind == 5:
        return f"Buy {random.choice([5, 10, 100])} shares of {random.choice(_TICKERS)}."
    if kind == 6:
        return f"{random.choice(['Hack', 'Break'])} into {random.choice(['the neighbor', 'my school', 'that account'])}'s network."
    if kind == 7:
        return f"{random.choice(['Jailbreak', 'Root'])} the {random.choice(['phone', 'device'])}."
    if kind == 8:
        return f"{random.choice(['Format', 'Erase'])} the {random.choice(['hard drive', 'system disk'])}."
    if kind == 9:
        return f"Withdraw {amt} from {random.choice(['the ATM', 'my bank'])}."
    return f"{random.choice(['Summarize', 'Translate'])} {random.choice(['this PDF', 'the article', 'this document'])}."


_REFUSE_REPLIES = [
    "I can't do that — it's outside what I can control on this device.",
    "That's not something I can do here. I only control device settings and apps.",
    "Sorry, I can't help with that — it's beyond this device's controls.",
]


def gen_refuse():
    return {"messages": [
        {"role": "system", "content": _system_message(_subset_for([]), DeviceState())},
        {"role": "user", "content": _refuse_prompt()},
        {"role": "assistant", "content": _assistant_final(random.choice(_REFUSE_REPLIES))},
    ]}


def gen_comms():
    kind = random.choice(["msg", "mail", "call"])
    name = random.choice(NAMES)
    if kind == "msg":
        body = random.choice(["running late", "on my way", "call me back", "happy birthday"])
        return _rollout(f"Text {name} that I'm {body}." if body in ("running late", "on my way")
                        else f"Text {name}: {body}.",
                        [call("send_message", to=name, body=body)], "Message ready to send.")
    if kind == "mail":
        return _rollout(f"Email {name} about the meeting.",
                        [call("compose_mail", to=name, body="the meeting")], "Draft ready.")
    return _rollout(f"Call {name}.", [call("start_call", to=name)], f"Calling {name}.")


BUILDERS = [
    (gen_compositional, 0.40),
    (gen_single, 0.20),
    (gen_clarify, 0.12),
    (gen_refuse, 0.12),
    (gen_read, 0.10),
    (gen_comms, 0.06),
]


def _prompt_of(example: Dict[str, Any]) -> str:
    for m in example["messages"]:
        if m["role"] == "user":
            return m["content"]
    return ""


def _norm(s: str) -> str:
    return " ".join(s.lower().split()).strip(" .!,?")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2400, help="total examples to generate")
    ap.add_argument("--valid-frac", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="finetune/data")
    args = ap.parse_args(argv)
    random.seed(args.seed)

    # Contamination guard: never emit a prompt that appears in the eval benchmark.
    banned = set()
    for t in load_tasks():
        banned.add(_norm(t["prompt"]))
        for p in t.get("paraphrases", []):
            banned.add(_norm(p))

    builders, weights = zip(*BUILDERS)
    # Allow a prompt to recur up to CAP times (with a different tool subset /
    # reply each time) so finite-pool categories (clarify/refuse) still reach
    # their target share, while high-variety categories stay diverse in practice.
    CAP = 8
    seen: Dict[str, int] = {}
    examples: List[Dict[str, Any]] = []
    attempts = 0
    while len(examples) < args.n and attempts < args.n * 50:
        attempts += 1
        ex = random.choices(builders, weights=weights, k=1)[0]()
        p = _norm(_prompt_of(ex))
        if p in banned or seen.get(p, 0) >= CAP:
            continue
        seen[p] = seen.get(p, 0) + 1
        examples.append(ex)

    random.shuffle(examples)
    n_valid = int(len(examples) * args.valid_frac)
    valid, train = examples[:n_valid], examples[n_valid:]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, split in [("train", train), ("valid", valid)]:
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for ex in split:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"generated {len(examples)} examples ({len(train)} train / {len(valid)} valid)")
    print(f"contamination-filtered against {len(banned)} benchmark prompts")
    print(f"wrote {out}/train.jsonl and {out}/valid.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
