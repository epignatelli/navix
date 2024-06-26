<p class="maiusc" style="margin-bottom: 0.8em;">A <b>fast</b>, fully <b>jittable, batched MiniGrid</b>  reimplemented in JAX for HIGH <b>THROUGHPUT</b></p>
<h1>Welcome to <b>NAVIX</b>!</h1>


**NAVIX** is a reimplementation of the [MiniGrid](https://minigrid.farama.org/) environment suite in JAX, and leverages JAX’s intermediate language representation to migrate the computation to different accelerators, such as GPUs and TPUs.

NAVIX is designed to be a drop-in replacement for the original MiniGrid environment, with the added benefit of being significantly faster.
Experiments that took **1 week**, now take **15 minutes**.

A [`navix.Environment`](api/environments/environment.md) is a [`flax.struct.PyTreeNode`](https://flax.readthedocs.io/en/latest/api_reference/flax.struct.html#flax.struct.PyTreeNode) and supports [`jax.vmap`](https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html), [`jax.jit`](https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html), [`jax.grad`](https://jax.readthedocs.io/en/latest/_autosummary/jax.grad.html), and all the other JAX's transformations.
See some examples [here](examples/getting_started.ipynb).

<br>
Most of the MiniGrid environments are supported, and the API is designed to be as close as possible to the original MiniGrid API.
However, some features might be missing, or the API might be slightly different.
If you find so, please [open an issue](https://github.com/epignatelli/navix/issues/new) or a [pull request](https://github.com/epignatelli/navix/pulls), contributions are welcome!


Thanks to JAX's backend, NAVIX offers:

- Multiple accelerators: NAVIX can run on CPU, GPU, or TPU.
- Performance boost: 200 000x speed up in batch mode or 20x unbatched mode.
- Parallellisation: NAVIX can run up to 2048 PPO agents (32768 environments!) in parallel on a single Nvidia A100 80Gb.
- Full automatic differentiation: NAVIX can compute gradients of the environment with respect to the agent's actions.


[Get started with NAVIX](examples/getting_started.ipynb){ .md-button .md-button--primary}