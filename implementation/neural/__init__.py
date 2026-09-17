"""Neural sequence layer and language model based on the memory-augmented
nonlinear Schrödinger field equation.

The mathematical equivalence between the equation's auxiliary-field memory
and the hidden state of structured state space models (S4, Mamba, RWKV) is
documented in `../../interfaces/06-state-space-models.md`. This package is
the concrete PyTorch implementation of that equivalence, plus a full
autoregressive language model built on top of it.

Modules
-------
layer        : The memory-NLS sequence layer (FFT-convolution implementation).
model        : MemoryNLSLanguageModel — the full autoregressive LM.
baselines    : TransformerLanguageModel — same-architecture baseline for A/B.
training     : Training infrastructure (AdamW + cosine schedule + AMP).
generation   : Sampling helpers including a character-level tokenizer.
"""

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

if _TORCH_AVAILABLE:
    from .layer import MemoryNLSLayer  # noqa: F401
    from .model import MemoryNLSConfig, MemoryNLSLanguageModel  # noqa: F401
    from .baselines import TransformerConfig, TransformerLanguageModel  # noqa: F401
    from .training import TrainConfig, train  # noqa: F401
    from .generation import CharTokenizer, generate_text  # noqa: F401
    __all__ = [
        "MemoryNLSLayer",
        "MemoryNLSConfig",
        "MemoryNLSLanguageModel",
        "TransformerConfig",
        "TransformerLanguageModel",
        "TrainConfig",
        "train",
        "CharTokenizer",
        "generate_text",
    ]
else:
    __all__ = []
    import warnings
    warnings.warn(
        "PyTorch is not installed. The neural sequence layer and language "
        "model are unavailable. Install with `pip install torch>=2.0`.",
        stacklevel=2,
    )
