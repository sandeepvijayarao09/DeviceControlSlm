"""Tests for the agent loop, parser, registry, and safety gate.

Run: cd python && python -m pytest -q   (or: python -m unittest -q)
These run anywhere - no model, no GPU - which is the whole point of the
mock backend (docs/design.md §3.4).
"""
from __future__ import annotations

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentcore import Agent, Sensitivity, ToolRegistry, parse_decision
from agentcore.backends import MockBackend
from tools import DeviceState, build_tools


def make_agent(scripted=None, policy=None, confirm=None):
    state = DeviceState(notes={"Home Wi-Fi": "wifi password is hunter2"})
    registry = ToolRegistry(build_tools(state))
    backend = MockBackend(scripted=scripted, policy=policy)
    agent = Agent(backend, registry, confirm=confirm)
    return agent, state


class TestParser(unittest.TestCase):
    def test_parses_action(self):
        d = parse_decision('{"thought":"t","action":{"tool":"start_timer","arguments":{"minutes":5}}}')
        self.assertTrue(d.is_tool_call)
        self.assertEqual(d.tool, "start_timer")
        self.assertEqual(d.arguments, {"minutes": 5})

    def test_parses_final(self):
        d = parse_decision('{"thought":"t","final_answer":"hi"}')
        self.assertTrue(d.is_final)
        self.assertEqual(d.final_answer, "hi")

    def test_extracts_json_from_prose(self):
        d = parse_decision('Sure! ```json\n{"final_answer": "ok"}\n``` hope that helps')
        self.assertEqual(d.final_answer, "ok")

    def test_flattened_tool_call(self):
        d = parse_decision('{"tool":"list_reminders","arguments":{}}')
        self.assertTrue(d.is_tool_call)
        self.assertEqual(d.tool, "list_reminders")

    def test_no_json(self):
        self.assertIsNotNone(parse_decision("no json here").parse_error)


class TestLoop(unittest.TestCase):
    def test_tool_then_final(self):
        agent, state = make_agent(
            scripted=[
                '{"action":{"tool":"start_timer","arguments":{"minutes":10}}}',
                '{"final_answer":"Timer set for 10 minutes."}',
            ]
        )
        res = agent.run("set a timer for 10 minutes")
        self.assertEqual(res.stopped_reason, "final_answer")
        self.assertEqual(state.timers[0]["minutes"], 10)
        self.assertEqual(res.answer, "Timer set for 10 minutes.")

    def test_invalid_args_rejected_then_repaired(self):
        # First call has wrong type; loop feeds error back and second call is valid.
        agent, state = make_agent(
            scripted=[
                '{"action":{"tool":"start_timer","arguments":{"minutes":"ten"}}}',
                '{"action":{"tool":"start_timer","arguments":{"minutes":10}}}',
                '{"final_answer":"done"}',
            ]
        )
        res = agent.run("timer")
        self.assertTrue(any(s.kind == "error" for s in res.steps))
        self.assertEqual(state.timers[0]["minutes"], 10)

    def test_unknown_tool_rejected(self):
        agent, _ = make_agent(
            scripted=[
                '{"action":{"tool":"nuke","arguments":{}}}',
                '{"final_answer":"cannot"}',
            ]
        )
        res = agent.run("do something")
        self.assertTrue(any(s.kind == "error" for s in res.steps))

    def test_confirm_gate_declined(self):
        # create_reminder is CONFIRM-sensitivity; declining blocks the side effect.
        agent, state = make_agent(
            scripted=[
                '{"action":{"tool":"create_reminder","arguments":{"text":"call mom","time":"6pm"}}}',
                '{"final_answer":"ok, cancelled"}',
            ],
            confirm=lambda name, args: False,
        )
        res = agent.run("remind me to call mom at 6pm")
        self.assertEqual(state.reminders, [])
        self.assertTrue(res.tool_calls[0].confirmation_required)

    def test_confirm_gate_approved(self):
        agent, state = make_agent(
            scripted=[
                '{"action":{"tool":"create_reminder","arguments":{"text":"call mom","time":"6pm"}}}',
                '{"final_answer":"done"}',
            ],
            confirm=lambda name, args: True,
        )
        agent.run("remind me to call mom at 6pm")
        self.assertEqual(len(state.reminders), 1)

    def test_max_steps_guard(self):
        # Never finishes -> loop must stop at max_steps, not run forever.
        agent, _ = make_agent(policy=lambda msgs: '{"action":{"tool":"list_reminders","arguments":{}}}')
        res = agent.run("loop forever")
        self.assertEqual(res.stopped_reason, "max_steps")


if __name__ == "__main__":
    unittest.main()
