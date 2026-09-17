"""Text generation helpers for the language models in this package.

The `MemoryNLSLanguageModel` and `TransformerLanguageModel` classes both
have built-in `generate` methods, so this module exists primarily for
char-level tokenization helpers that pair with the training experiments.
"""

from __future__ import annotations

from typing import Optional

import torch


class CharTokenizer:
    """Minimal byte-level character tokenizer.

    Builds a vocabulary of unique characters from a training text, with
    optional special tokens. Used by the TinyShakespeare experiment.
    """

    def __init__(self, text: str, special_tokens: Optional[list[str]] = None):
        special_tokens = special_tokens or []
        chars = sorted(set(text))
        self.special_tokens = special_tokens
        self.id_to_char = {i: c for i, c in enumerate(special_tokens + chars)}
        self.char_to_id = {c: i for i, c in self.id_to_char.items()}
        self.vocab_size = len(self.id_to_char)

    def encode(self, text: str) -> list[int]:
        return [self.char_to_id[c] for c in text if c in self.char_to_id]

    def decode(self, ids) -> str:
        return "".join(self.id_to_char[i] for i in ids if i in self.id_to_char)


@torch.no_grad()
def generate_text(
    model,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 40,
    device: str = "cuda",
) -> str:
    """Sample text from a model trained with the given tokenizer.

    Args:
        model: a model with a `.generate(input_ids, max_new_tokens, ...)` method.
        tokenizer: CharTokenizer used during training.
        prompt: starting text.
        max_new_tokens: how many tokens to sample.
        temperature: sampling temperature.
        top_k: restrict to top-k tokens.
        device: device to run on.

    Returns:
        Generated text (including the prompt).
    """
    model.eval()
    prompt_ids = tokenizer.encode(prompt) or [0]
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    return tokenizer.decode(out[0].tolist())
