# Implementation overview

This page describes the code included in this extraction. It provides a local
entry point for readers without requiring the larger project's documents.

## Field dynamics

The equation stated in the solver is:

$$
i\hbar\,\partial_t\Psi =
\left[-\frac{\hbar^2}{2m}\nabla^2 + V_{\mathrm{ext}} + \Lambda|\Psi|^2
+ V_{\mathrm{mem}} + \alpha(-\Delta)^{\sigma/2} - i\Gamma\right]\Psi + \eta.
$$

The solver stores the field and auxiliary memory fields on a periodic 3D lattice.
It uses split-step propagation with Fourier-space kinetic evolution. The memory
potential is assembled from auxiliary fields whose relaxation rates and coupling
weights are configured in `MemoryConfig`. Spatial kernels and observables live
in separate modules; the backend selects NumPy or CuPy.

Start with [solver_3d.py](../implementation/physics/solver_3d.py),
[kernels.py](../implementation/physics/kernels.py) and
[observables.py](../implementation/physics/observables.py). Parameter choices
used by each script are preserved in `experiments/physics/`.

## Neural sequence model

The neural implementation translates an auxiliary-memory structure into a
sequence-mixing layer. It does not run the spatial field solver during training.

For a projected token representation `x_in`, the layer computes a scalar density
per position, `rho = mean(x_in**2)`. Each memory mode has a fixed exponential
kernel with `alpha_j = exp(-nu_j * dt)`. A causal convolution gives the sequence
of auxiliary states. Their weighted sum forms a memory potential, which combines
with the instantaneous cubic interaction and multiplies `x_in` before output
projection and a residual addition. Optional stochastic forcing is implemented
in the layer; its settings come from the experiment configuration.

The convolution uses zero-padded real FFTs. Relaxation rates and memory couplings
are registered buffers; the input and output projections are trainable. The
language model stacks these mixers with normalization and feed-forward layers,
between token embeddings and a tied output head.

Entry points:

- [layer.py](../implementation/neural/layer.py): kernel, causal convolution, interaction.
- [model.py](../implementation/neural/model.py): configuration, blocks, language model.
- [baselines.py](../implementation/neural/baselines.py): causal Transformer implementation.
- [training.py](../implementation/neural/training.py): shared training helper.
- [scale_up_dynamics.py](../experiments/neural/scale_up_dynamics.py): dedicated enwik8 loop.

The long-horizon scripts define their own training loops. Updating the shared
helper would not automatically update those experiments. The enwik8 comparison
uses FFN multipliers 5 and 4 for Memory-NLS and Transformer respectively; their
parameter counts are close, but not identical. Memory-NLS disables positional
embeddings in that script; the Transformer uses learned positional embeddings.

The preserved Memory-NLS block passes its normalized input to a mixer that adds
its own residual. This differs from retaining the unnormalized block input in
the residual path. It is part of the reference implementation and has
not been changed during publication preparation.
