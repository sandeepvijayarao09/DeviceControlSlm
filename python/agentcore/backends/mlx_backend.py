"""MLX backend — real on-device / on-Mac inference via mlx-lm.

Runs natively on Apple Silicon (Mac target, or a physical iPhone/iPad through
the Swift port). It is imported lazily so the rest of the package works on
Linux/CI where `mlx_lm` is not installed. See docs/design.md §3.

Install on a Mac:  pip install mlx-lm
Default model:     mlx-community/Qwen2.5-3B-Instruct-4bit
"""
from __future__ import annotations

from typing import List, Optional

from .base import GenerationConfig, Message, ModelBackend

DEFAULT_MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"


class MLXBackend(ModelBackend):
    name = "mlx"

    def __init__(self, model_path: str = DEFAULT_MODEL) -> None:
        try:
            from mlx_lm import load  # type: ignore
        except ImportError as e:  # pragma: no cover - only hit off-Mac
            raise ImportError(
                "mlx-lm is required for MLXBackend. Install it on a Mac with "
                "`pip install mlx-lm`. On Linux/CI use MockBackend instead."
            ) from e
        self.model_path = model_path
        self._model, self._tokenizer = load(model_path)

    def generate(self, messages: List[Message], config: GenerationConfig) -> str:
        from mlx_lm import generate as mlx_generate  # type: ignore
        from mlx_lm.sample_utils import make_sampler  # type: ignore

        prompt = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        sampler = make_sampler(temp=config.temperature)
        return mlx_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=config.max_tokens,
            sampler=sampler,
            verbose=False,
        )
