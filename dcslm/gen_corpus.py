"""Corpus generator for the from-scratch, device-control-ONLY SLM.

A from-scratch tiny model has no borrowed language knowledge, so its coverage
equals our data's coverage. The whole strategy therefore rests on *phrasing
diversity*: many natural surface forms per command. Because the task is only
device control, the OUTPUT is a tiny closed set, expressed as a compact DSL the
model can actually learn:

    request  ->  "set_wifi|enabled=false ; set_brightness|level=30"
    request  ->  "ASK"        (under-specified -> clarify)
    request  ->  "REFUSE"     (out of scope)

Each corpus line is:  \x01<request>\x02<dsl>\x03\n
(\x01 start, \x02 command marker, \x03 end) — single chars so a char-level LM
learns the example boundaries. The same delimiters are used by dcslm/backend.py
at inference.

    python -m dcslm.gen_corpus --n 16000 --out dcslm/data/corpus.txt
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from bench.dataset import load_tasks
from bench.normalize import norm_time_24h

U, C, E = "\x01", "\x02", "\x03"


def _norm(s: str) -> str:
    return " ".join(s.lower().split()).strip(" .!,?")


def benchmark_prompts() -> set:
    """Every DeviceControlBench prompt + paraphrase — never allowed into training."""
    banned = set()
    for t in load_tasks():
        banned.add(_norm(t["prompt"]))
        for p in t.get("paraphrases", []):
            banned.add(_norm(p))
    return banned

# ---- phrasing banks ---------------------------------------------------------
ON = ["turn on", "enable", "switch on", "activate", "put on", "flip on", "get", "bring up",
      "fire up", "power on", "pop on", "kick on", "toggle on"]
OFF = ["turn off", "disable", "switch off", "deactivate", "kill", "shut off", "flip off",
       "cut", "cut off", "drop", "power down", "toggle off", "knock off"]
POLITE = ["", "", "", "please ", "can you ", "could you ", "hey, ", "i need you to ",
          "would you ", "mind ", "let's ", "go ahead and ", "i'd like you to "]
TAIL = ["", "", "", " please", " for me", " now", " thanks", " real quick", " if you can",
        " right now", " already"]

TOGGLES = {
    "set_wifi": ["wifi", "wi-fi", "the wifi", "wireless"],
    "set_bluetooth": ["bluetooth", "the bluetooth", "bt"],
    "set_airplane_mode": ["airplane mode", "flight mode", "aeroplane mode"],
    "set_cellular_data": ["cellular data", "mobile data", "cellular", "data"],
    "set_hotspot": ["the hotspot", "personal hotspot", "hotspot", "tethering"],
    "set_do_not_disturb": ["do not disturb", "dnd", "do-not-disturb"],
    "set_dark_mode": ["dark mode", "dark theme"],
    "set_orientation_lock": ["rotation lock", "orientation lock", "portrait lock"],
    "set_mute": ["mute", "silent mode"],
    "set_low_power_mode": ["low power mode", "battery saver", "power saving mode", "low power"],
    "set_location_services": ["location services", "location", "gps"],
    "set_flashlight": ["the flashlight", "flashlight", "the torch", "the light"],
}
APPS = ["Spotify", "Safari", "Notes", "Mail", "Maps", "Calendar", "Photos", "Messages",
        "Music", "Slack", "Chrome", "Weather", "Camera", "Settings", "the App Store",
        "Instagram", "WhatsApp", "Files", "Reminders", "Clock"]
NAMES = ["Alex", "Sam", "Jordan", "Taylor", "mom", "dad", "Chris", "Riley", "my boss", "Priya"]
TASKS = ["buy milk", "call the dentist", "email the report", "water the plants",
         "take out the trash", "pick up the kids", "pay rent", "book a table", "stretch"]
FOCUS = ["work", "personal", "sleep", "driving", "fitness"]
RINGER = ["ring", "vibrate", "silent"]


def _cap(s):
    return s[0].upper() + s[1:] if s and random.random() < 0.6 else s


def _wrap(frag):
    """Add polite prefixes / tails / capitalization to a bare imperative fragment."""
    s = random.choice(POLITE) + frag + random.choice(TAIL)
    s = _cap(s)
    if random.random() < 0.5:
        s += random.choice([".", "", "!", "."])
    return s


def _time():
    h, m = random.randint(1, 12), random.choice([0, 0, 15, 30, 45])
    ap = random.choice(["am", "pm"])
    surface = random.choice([f"{h}{ap}", f"{h}:{m:02d}{ap}", f"{h} {ap}", f"{h}:{m:02d} {ap}"]) if m or random.random() < .5 else f"{h}{ap}"
    return surface, norm_time_24h(surface)


# ---- atomic action generators: return (fragment, "tool|k=v|..") -------------
def g_toggle():
    tool, names = random.choice(list(TOGGLES.items()))
    name = random.choice(names)
    on = random.random() < 0.5
    verb = random.choice(ON if on else OFF)
    templates = [f"{verb} {name}", f"{name} {'on' if on else 'off'}",
                 f"i want {name} {'on' if on else 'off'}", f"{verb} {name}"]
    return random.choice(templates), f"{tool}|enabled={'true' if on else 'false'}"


def g_level():
    which, tool = random.choice([("brightness", "set_brightness"), ("volume", "set_volume")])
    n = random.choice([0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 100])
    templates = [f"set the {which} to {n}%", f"{which} to {n}", f"set {which} {n}%",
                 f"change the {which} to {n} percent", f"put the {which} at {n}",
                 f"make the {which} {n}%", f"turn the {which} {'down' if n<40 else 'up'} to {n}%"]
    return random.choice(templates), f"{tool}|level={n}"


def g_alarm():
    s, norm = _time()
    templates = [f"set an alarm for {s}", f"wake me at {s}", f"alarm at {s}", f"set alarm {s}",
                 f"wake me up at {s}"]
    return random.choice(templates), f"set_alarm|time={norm}"


def g_timer():
    n = random.choice([1, 2, 3, 5, 8, 10, 15, 20, 25, 30, 45, 60, 90])
    templates = [f"set a timer for {n} minutes", f"timer for {n} min", f"start a {n} minute timer",
                 f"{n} minute timer", f"set a {n} min timer"]
    return random.choice(templates), f"start_timer|minutes={n}"


def g_app():
    a = random.choice(APPS)
    if random.random() < 0.8:
        verb = random.choice(["open", "launch", "start", "fire up"])
        return f"{verb} {a}", f"open_app|name={a.lower()}"
    return f"close {a}", f"close_app|name={a.lower()}"


def g_media():
    act, frags = random.choice([
        ("pause", ["pause the music", "pause"]),
        ("play", ["resume", "resume playback", "play the music"]),
        ("next", ["skip this song", "next track", "skip"]),
        ("previous", ["go back a track", "previous song"]),
        ("stop", ["stop the music", "stop playback"]),
    ])
    return random.choice(frags), f"media_control|action={act}"


def g_focus():
    m = random.choice(FOCUS)
    return random.choice([f"set my focus to {m}", f"turn on {m} focus", f"{m} focus", f"switch to {m} focus"]), \
        f"set_focus_mode|mode={m}"


def g_ringer():
    m = random.choice(RINGER)
    return random.choice([f"set the ringer to {m}", f"put the phone on {m}", f"switch to {m}"]), \
        f"set_ringer_mode|mode={m}"


def g_screenshot():
    return random.choice(["take a screenshot", "screenshot", "grab a screenshot", "capture the screen"]), \
        "take_screenshot"


def g_reminder():
    t = random.choice(TASKS)
    if random.random() < 0.6:
        s, norm = _time()
        return random.choice([f"remind me to {t} at {s}", f"set a reminder to {t} at {s}"]), \
            f"create_reminder|text={t}|time={norm}"
    return random.choice([f"remind me to {t}", f"add a reminder to {t}"]), f"create_reminder|text={t}"


def g_power():
    if random.random() < 0.5:
        return random.choice(["restart the phone", "reboot", "restart the device", "restart"]), "restart_device"
    return random.choice(["shut down", "power off", "turn off the device", "shut down the phone"]), "shutdown_device"


def g_comms():
    n = random.choice(NAMES)
    r = random.random()
    if r < 0.5:
        body = random.choice(["running late", "on my way", "call me", "happy birthday", "i quit"])
        return random.choice([f"text {n} that i'm {body}", f"message {n} saying {body}", f"text {n}: {body}"]), \
            f"send_message|to={n}|body={body}"
    if r < 0.75:
        return f"email {n} about the meeting", f"compose_mail|to={n}|body=the meeting"
    return random.choice([f"call {n}", f"give {n} a call", f"phone {n}"]), f"start_call|to={n}"


# Indirect intents: situational phrasing -> the command a person means. Teaches the
# model to handle real requests that never name the setting explicitly.
INTENTS = [
    (["i'm going to bed", "time for bed", "heading to sleep", "going to sleep now",
      "off to bed", "calling it a night"], "set_focus_mode|mode=sleep"),
    (["i'm getting on a plane", "boarding now", "about to take off", "getting on my flight",
      "we're taking off"], "set_airplane_mode|enabled=true"),
    (["save my battery", "my battery is dying", "battery's low", "need to conserve battery",
      "running low on juice"], "set_low_power_mode|enabled=true"),
    (["i'm driving now", "about to drive", "getting in the car", "starting my commute"],
     "set_focus_mode|mode=driving"),
    (["i'm at the gym", "starting my workout", "time to work out", "heading to the gym"],
     "set_focus_mode|mode=fitness"),
    (["i'm in a meeting", "heading into a meeting", "don't disturb me", "hold my notifications"],
     "set_do_not_disturb|enabled=true"),
    (["it's too bright", "the screen is blinding me", "too much glare", "ease the brightness"],
     "set_brightness|level=20"),
    (["i can't see the screen", "the screen's too dark", "brighten things up"],
     "set_brightness|level=90"),
    (["i'm back", "i'm done with my meeting", "turn notifications back on"],
     "set_do_not_disturb|enabled=false"),
    (["it's dark in here", "i need some light", "light it up"], "set_flashlight|enabled=true"),
]


def g_intent():
    frags, cmd = random.choice(INTENTS)
    return random.choice(frags), cmd


ACTIONS = [g_toggle, g_level, g_alarm, g_timer, g_app, g_media, g_focus, g_ringer,
           g_screenshot, g_reminder, g_power, g_comms, g_intent]

# actions safe to chain in a compositional request (avoid power/comms noise)
CHAINABLE = [g_toggle, g_level, g_alarm, g_timer, g_app, g_media, g_focus, g_ringer,
             g_screenshot, g_reminder]

CONNECTORS = [" and ", ", then ", ", also ", ", and ", " and then "]

CLARIFY = ["turn it on", "turn that off", "set the brightness", "set an alarm", "open the app",
           "remind me", "set a timer", "change the volume", "text them", "call", "make it brighter",
           "set my focus", "adjust the volume", "open it", "set it", "wake me up", "play something",
           "set the ringer", "launch it", "schedule it"]

REFUSE = ["transfer $500 to my landlord", "order a pizza", "book a flight to paris",
          "post this to instagram", "buy 10 shares of apple", "jailbreak my phone",
          "withdraw cash from my bank", "hack the neighbor's wifi", "translate this page",
          "summarize this pdf", "mine some bitcoin", "delete the operating system",
          "wire money to this account", "sell my tesla stock", "reserve a hotel in tokyo"]

READS = [
    (["how much battery do i have", "battery level", "what's my battery at"], "get_battery"),
    (["how much storage is left", "free storage", "storage space"], "get_storage"),
    (["am i on wifi", "am i connected", "connection status"], "get_connectivity_status"),
]


def gen_one():
    r = random.random()
    if r < 0.35:                                   # compositional
        k = random.choice([2, 2, 3])
        used, frags, cmds = set(), [], []
        while len(cmds) < k:
            frag, cmd = random.choice(CHAINABLE)()
            tool = cmd.split("|")[0]
            if tool in used:
                continue
            used.add(tool)
            frags.append(frag)
            cmds.append(cmd)
        req = random.choice(CONNECTORS).join(frags)
        return _wrap(req), " ; ".join(cmds)
    if r < 0.78:                                   # single action
        frag, cmd = random.choice(ACTIONS)()
        return _wrap(frag), cmd
    if r < 0.86:                                   # clarify
        return _wrap(random.choice(CLARIFY)), "ASK"
    if r < 0.94:                                   # refuse
        return _wrap(random.choice(REFUSE)), "REFUSE"
    frags, tool = random.choice(READS)             # read
    return _wrap(random.choice(frags)), tool


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--out", default="dcslm/data/corpus.txt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    random.seed(args.seed)
    banned = benchmark_prompts()                   # contamination guard vs the eval set

    lines, seen, skipped = [], {}, 0
    while len(lines) < args.n:
        req, dsl = gen_one()
        key = _norm(req)
        if key in banned:                          # never train on a benchmark prompt
            skipped += 1
            continue
        if seen.get(key, 0) >= 6:                  # cap repeats, keep diversity
            continue
        seen[key] = seen.get(key, 0) + 1
        lines.append(f"{U}{req}{C}{dsl}{E}")
    random.shuffle(lines)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    uniq = len(seen)
    print(f"wrote {len(lines)} examples ({uniq} unique prompts) to {out}")
    print(f"contamination guard: skipped {skipped} generations colliding with {len(banned)} benchmark prompts")
    leaks = benchmark_prompts() & set(seen)
    print(f"benchmark prompts leaked into corpus: {len(leaks)} (must be 0)")
    print(f"chars: {sum(len(l) for l in lines)} | avg line len {sum(len(l) for l in lines)//len(lines)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
