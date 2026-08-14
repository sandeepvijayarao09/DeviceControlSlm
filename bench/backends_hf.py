"""HuggingFace Transformers backend (CPU/GPU), used for the portable *quality*
evaluation. Standard, reproducible, and hardware-agnostic; the Apple-Silicon
*efficiency* numbers come separately from the MLX backend (eval-protocol §7).

The model loads once and is reused across tasks. `tool`-role observations are
mapped to a user turn since not every chat template defines a tool role.
"""
from __future__ import annotations

from typing import List

from agentcore.backends.base import GenerationConfig, Message, ModelBackend


class HFBackend(ModelBackend):
    name = "hf"

    def __init__(self, model_id: str, dtype: str = "float32", device: str = "cpu") -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=getattr(torch, dtype), device_map=device
        )
        self.model.eval()
        self.device = device
        # record the resolved revision for the run manifest
        self.revision = getattr(self.model.config, "_commit_hash", None) or "unknown"

    def _prep(self, messages: List[Message]) -> List[Message]:
        out = []
        for m in messages:
            if m["role"] == "tool":
                out.append({"role": "user", "content": f"[tool result]\n{m['content']}"})
            else:
                out.append(m)
        return out

    def generate(self, messages: List[Message], config: GenerationConfig) -> str:
        torch = self._torch
        text = self.tokenizer.apply_chat_template(
            self._prep(messages), add_generation_prompt=True, tokenize=False
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        gen_kwargs = dict(max_new_tokens=config.max_tokens, pad_token_id=self.tokenizer.eos_token_id)
        if config.temperature and config.temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=config.temperature)
        else:
            gen_kwargs.update(do_sample=False)
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
