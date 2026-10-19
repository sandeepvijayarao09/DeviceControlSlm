"""Benchmark-integrity tests: dataset loads, is balanced, and every task's gold
behaviour passes its own checker (gold replay). These guard the benchmark the
same way unit tests guard the agent loop.
"""
from __future__ import annotations

import os
import sys
import unittest

# make both `agentcore` (python/) and `bench` (repo root) importable
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "python"))

from bench.dataset import CATEGORY_ORDER, load_tasks
from bench.gold_replay import validate
from bench.metrics import normalize_args


class TestBenchmarkIntegrity(unittest.TestCase):
    def test_loads_and_balanced(self):
        tasks = load_tasks()
        self.assertGreaterEqual(len(tasks), 48)
        by_cat = {c: 0 for c in CATEGORY_ORDER}
        for t in tasks:
            by_cat[t["category"]] += 1
        for c, n in by_cat.items():
            self.assertGreater(n, 0, f"category {c} is empty")

    def test_unique_ids(self):
        ids = [t["id"] for t in load_tasks()]
        self.assertEqual(len(ids), len(set(ids)))

    def test_gold_replay_all_consistent(self):
        failures = validate()
        self.assertEqual(failures, [], f"gold-inconsistent tasks: {failures}")


class TestNormalization(unittest.TestCase):
    def test_time_and_duration(self):
        self.assertEqual(normalize_args({"time": "6pm"})["time"], "18:00")
        self.assertEqual(normalize_args({"time": "quarter past 9"})["time"], "09:15")
        self.assertEqual(normalize_args({"minutes": "ten"})["minutes"], 10)
        self.assertEqual(normalize_args({"minutes": "1 hour"})["minutes"], 60)

    def test_text_casefold(self):
        self.assertEqual(normalize_args({"to": "  Alex "})["to"], "alex")


if __name__ == "__main__":
    unittest.main()
