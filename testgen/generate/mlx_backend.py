"""Generation with mlx-lm under fixed decoding settings (D010, FT14, FT16).

- Thinking is disabled for every model that supports the switch, at the
  chat-template level, so no reasoning tokens consume the budget.
- Greedy decoding (temperature 0) for the headline numbers.
- ``generate_many`` batches prompts through mlx-lm's native batch generator;
  results are returned in input order with per-item token counts and timing.

mlx-lm >= 0.31.1 is required: 0.31.0 had batched KV-cache cross-contamination.
"""

import re
import time
from dataclasses import dataclass

import mlx_lm
from mlx_lm.sample_utils import make_sampler

from config import MAX_NEW_TOKENS

# The template opens <think> inside the prompt, so output holds only the closing tag.
_THINK = re.compile(r"^(?:<think>)?.*?</think>\s*", re.S)


@dataclass
class Generation:
    text: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float
    thinking_tokens: int = 0  # tokens inside <think>...</think>, stripped from text


class Backend:
    def __init__(
        self, model_id: str, adapter_path: str | None = None, thinking: bool = False
    ) -> None:
        """``adapter_path``: a LoRA adapter directory (mlx-lm format) applied on load.
        ``thinking``: enable the model's reasoning block; it is stripped from the
        returned text and counted in ``thinking_tokens`` (still inside the budget)."""
        self.model_id = model_id
        self.adapter_path = adapter_path
        self.thinking = thinking
        self.model, self.tokenizer = mlx_lm.load(model_id, adapter_path=adapter_path)

    def _encode(self, messages: list[dict[str, str]]) -> list[int]:
        kwargs = {"add_generation_prompt": True, "tokenize": True}
        try:
            return self.tokenizer.apply_chat_template(
                messages, enable_thinking=self.thinking, **kwargs
            )
        except TypeError:  # template without a thinking switch (e.g. Qwen2.5-Coder)
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    def generate_many(
        self,
        batch: list[list[dict[str, str]]],
        max_tokens: int = MAX_NEW_TOKENS,
        temperature: float = 0.0,
    ) -> list[Generation]:
        prompts = [self._encode(m) for m in batch]
        start = time.monotonic()
        resp = mlx_lm.batch_generate(
            self.model,
            self.tokenizer,
            prompts,
            max_tokens=max_tokens,
            sampler=make_sampler(temp=temperature),
        )
        elapsed = time.monotonic() - start
        out = []
        for p, text in zip(prompts, resp.texts, strict=True):
            n_out = len(self.tokenizer.encode(text))
            n_think = 0
            if self.thinking:
                stripped = _THINK.sub("", text, count=1)
                n_think = n_out - len(self.tokenizer.encode(stripped))
                text = stripped
            out.append(Generation(text, len(p), n_out, elapsed / len(batch), n_think))
        return out
