"""Backend that runs the from-scratch device-control SLM on DeviceControlBench.

The model emits a compact DSL in one shot (e.g. "set_wifi|enabled=false ;
set_brightness|level=30", or "ASK" / "REFUSE"). The agent loop, however, wants
ONE decision per turn. This adapter bridges the two: on the first turn it plans
(runs the model once, parses the DSL), then replays the plan one command per
turn and finishes — so the same loop, sandbox, and checkers score it exactly
like any other model. No tool schemas are put in the prompt: a device-only
model has the command set baked in from training.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from agentcore.backends.base import GenerationConfig, Message, ModelBackend
from scratch.model import GPT, GPTConfig

U, C, E = "\x01", "\x02", "\x03"


def _typed(v: str):
    if v == "true":
        return True
    if v == "false":
        return False
    if v.lstrip("-").isdigit():
        return int(v)
    return v


def parse_dsl(dsl: str) -> Tuple[str, List[Dict]]:
    """-> (kind, actions) where kind in {actions, ask, refuse}."""
    dsl = dsl.strip()
    if dsl == "ASK":
        return "ask", []
    if dsl == "REFUSE":
        return "refuse", []
    actions = []
    for part in dsl.split(";"):
        part = part.strip()
        if not part:
            continue
        fields = part.split("|")
        tool = fields[0].strip()
        args = {}
        for kv in fields[1:]:
            if "=" in kv:
                k, v = kv.split("=", 1)
                args[k.strip()] = _typed(v.strip())
        if tool:
            actions.append({"tool": tool, "arguments": args})
    return ("actions", actions) if actions else ("refuse", [])


class ScratchDeviceBackend(ModelBackend):
    name = "scratch"
    revision = "dcslm-scratch-v1"

    def __init__(self, ckpt_path: str):
        ck = torch.load(Path(ckpt_path), map_location="cpu", weights_only=False)
        self.model = GPT(GPTConfig(**ck["cfg"]))
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        self.stoi, self.itos = ck["stoi"], ck["itos"]
        self.block = ck["cfg"]["block_size"]
        self._plan_cache: Dict[str, Tuple[str, List[Dict]]] = {}

    @torch.no_grad()
    def _decode_plan(self, request: str) -> Tuple[str, List[Dict]]:
        prompt = U + request + C
        ids = [self.stoi[c] for c in prompt if c in self.stoi]
        idx = torch.tensor([ids], dtype=torch.long)
        end_id = self.stoi[E]
        out = []
        for _ in range(96):                        # greedy decode until <end>
            logits, _ = self.model(idx[:, -self.block:])
            nxt = int(torch.argmax(logits[0, -1]))
            if nxt == end_id:
                break
            out.append(nxt)
            idx = torch.cat([idx, torch.tensor([[nxt]])], dim=1)
        dsl = "".join(self.itos[i] for i in out)
        return parse_dsl(dsl)

    def _plan(self, request: str) -> Tuple[str, List[Dict]]:
        if request not in self._plan_cache:
            self._plan_cache[request] = self._decode_plan(request)
        return self._plan_cache[request]

    def generate(self, messages: List[Message], config: GenerationConfig) -> str:
        request = next((m["content"] for m in messages if m["role"] == "user"), "")
        kind, actions = self._plan(request)
        if kind == "ask":
            return json.dumps({"thought": "under-specified", "final_answer":
                               "Could you clarify which one, or the value you want?"})
        if kind == "refuse":
            return json.dumps({"thought": "out of scope", "final_answer":
                               "I can't do that on this device."})
        n_done = sum(1 for m in messages if m["role"] == "assistant")
        if n_done < len(actions):
            a = actions[n_done]
            return json.dumps({"thought": "device command", "action": a})
        return json.dumps({"thought": "done", "final_answer": "Done."})
