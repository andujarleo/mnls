# Memory-Nonlinear State Models


---

![Neural training trajectory](assets/scale_up_val_ppl.png)

*anti-collapse mechanism in optimization dynamics: at 70M parameters
on enwik8, Memory-NLS descends monotonically to a stable plateau; Transformer
without the structural mechanism crashes catastrophically at step 28000 and
never fully recovers. The structural form is operative across substrates as
different as 3D field dynamics and neural network optimization.*

---

**Optimization-dynamics anti-collapse** (70M parameters, enwik8, 50,000 training steps):

| Quantity | Memory-NLS | Transformer |
|---|---|---|
| Final val perplexity | 4.27 | 4.87 |
| Min val perplexity | 3.86 (step 48000) | 2.54 (step 22500) |
| Catastrophic collapse | None | Step 28000–34000, peak ppl 27.17 |
| Trajectory shape | Monotonic descent + plateau | Descent → crash → partial recovery |


---


All results use fixed random seeds and reproduce bit-for-bit on identical hardware (NVIDIA RTX 4060 Laptop GPU, Arch Linux, CUDA 12.x).

