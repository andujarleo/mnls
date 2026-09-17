"""Small CPU execution checks, independent of the recorded training runs."""
import unittest

import torch

from implementation.neural.layer import _causal_conv1d_fft
from implementation.neural import MemoryNLSConfig, MemoryNLSLanguageModel


class NeuralChecks(unittest.TestCase):
    def test_fft_matches_direct_causal_convolution(self):
        rho = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
        kernel = torch.tensor([[1.0], [0.5], [0.25], [0.125]])
        expected = torch.tensor([[[1.0], [2.5], [4.25], [6.125]]])
        torch.testing.assert_close(_causal_conv1d_fft(rho, kernel), expected)

    def test_language_model_backward_and_causality(self):
        torch.manual_seed(42)
        model = MemoryNLSLanguageModel(MemoryNLSConfig(
            vocab_size=16, d_model=16, n_layers=2, n_heads=2, max_seq_len=8,
        ))
        tokens = torch.arange(8).unsqueeze(0)
        logits, loss = model(tokens, (tokens + 1) % 16)
        self.assertEqual(tuple(logits.shape), (1, 8, 16))
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all().item())
        changed = tokens.clone()
        changed[:, 4:] = 15
        with torch.no_grad():
            other, _ = model(changed)
        torch.testing.assert_close(logits[:, :4], other[:, :4], atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
