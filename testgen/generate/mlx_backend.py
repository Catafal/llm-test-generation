"""Generation with mlx-lm under fixed decoding settings (D010, FT14, FT16).

- Thinking is disabled for every model that supports the switch, at the
  chat-template level, so no reasoning tokens consume the budget.
- Greedy decoding (temperature 0) for the headline numbers.
- ``generate_many`` batches prompts through mlx-lm's native batch generator;
  results are returned in input order with per-item token counts and timing.

mlx-lm >= 0.31.1 is required: 0.31.0 had batched KV-cache cross-contamination.
"""

import time
from dataclasses import dataclass

import mlx_lm
from mlx_lm.sample_utils import make_sampler

from config import MAX_NEW_TOKENS


@dataclass
class Generation:
    text: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float


class Backend:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.model, self.tokenizer = mlx_lm.load(model_id)

    def _encode(self, messages: list[dict[str, str]]) -> list[int]:
        kwargs = {"add_generation_prompt": True, "tokenize": True}
        try:
            return self.tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
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
            out.append(Generation(text, len(p), n_out, elapsed / len(batch)))
        return out
