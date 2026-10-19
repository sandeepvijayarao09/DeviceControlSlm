"""Interactive CLI demo of the on-device agent.

Default uses the rule-based reference policy so it runs anywhere:
    cd python && python demo.py
    python demo.py "remind me to call mom at 6pm"

On a Mac with mlx-lm installed, use the real model:
    python demo.py --backend mlx "set a timer for 5 minutes"
"""
from __future__ import annotations

import argparse
import sys

from agentcore import Agent, ToolRegistry
from eval.reference_policy import reference_policy
from tools import DeviceState, build_tools


def _confirm(name: str, args: dict) -> bool:
    ans = input(f"  [confirm] run {name}({args})? [Y/n] ").strip().lower()
    return ans in ("", "y", "yes")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("request", nargs="*", help="the request; omit for a prompt loop")
    parser.add_argument("--backend", default="reference", choices=["reference", "mlx"])
    parser.add_argument("--yes", action="store_true", help="auto-approve confirmations")
    args = parser.parse_args()

    state = DeviceState(notes={"Home Wi-Fi": "wifi password is hunter2"})
    registry = ToolRegistry(build_tools(state))

    if args.backend == "mlx":
        from agentcore.backends.mlx_backend import MLXBackend

        backend = MLXBackend()
    else:
        from agentcore.backends import MockBackend

        backend = MockBackend(policy=reference_policy)

    confirm = (lambda n, a: True) if args.yes else _confirm
    agent = Agent(backend, registry, confirm=confirm)

    def handle(request: str) -> None:
        print(f"\n> {request}")
        result = agent.run(request)
        for step in result.steps:
            if step.kind == "tool_call":
                print(f"  · called {step.tool}({step.arguments}) -> {step.observation}")
            elif step.kind == "error":
                print(f"  · format/validation error: {step.observation}")
        print(f"  = {result.answer}   [{result.stopped_reason}]")

    if args.request:
        handle(" ".join(args.request))
        return 0

    print("DeviceControlSlm demo. Type a request, or 'quit'.")
    while True:
        try:
            line = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if line.lower() in ("quit", "exit"):
            break
        if line:
            handle(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
