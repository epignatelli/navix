# Copyright 2023 The Navix Authors.

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Evolution-Strategies primitives shared by every navix hyperparameter
search: `Experiment.run_hparam_search` (navix/experiment.py, searches a
navix `Agent`'s own `HParams`) and `navix.benchmarks.search.
search_hparams` (searches an arbitrary `AlgorithmEntry`-shaped
trainable's hyperparameters, e.g. an external library like rejax). Both
follow the same OpenAI-ES (Salimans et al., 2017 - https://arxiv.org/
abs/1703.03864) shape - antithetic Gaussian sampling around a per-field-
scaled mean - only how a generation's population actually gets trained
and scored differs, which is why that part isn't shared here."""
from typing import Dict, Tuple

import distrax
import jax
import jax.numpy as jnp


class LogUniform(distrax.Uniform):
    """Log-uniform over `[low, high]` (both `> 0`): `log(sample)` is
    uniform in `[log(low), log(high))`, so every order of magnitude in
    the range gets equal sampling weight - the right choice for a field
    like a learning rate, where the right order of magnitude matters
    far more than the right value within one. A plain `distrax.
    Uniform(low, high)` over several orders of magnitude does not do
    this: `Uniform(1e-5, 1e-2)` puts ~90% of its mass above `1e-3`,
    starving the smaller end where a value like PPO's own `2.5e-4`
    default typically lives - confirmed in practice, searching navix's
    own PPO with a plain `Uniform(1e-5, 1e-2)` for `lr` found values
    clustered at `1e-3`-`1e-2` across every environment, nowhere near
    the known-good region."""

    def __init__(self, low: float, high: float):
        super().__init__(low=jnp.log(low), high=jnp.log(high))

    def sample(self, *, seed, sample_shape=()):
        return jnp.exp(super().sample(seed=seed, sample_shape=sample_shape))

    def sample_n(self, rng, n):
        return jnp.exp(super().sample_n(rng, n))


def probe_hparam_field_stats(
    hparams_distr: Dict[str, distrax.Distribution], n_probe: int, key: jax.Array
) -> Tuple[Dict[str, jax.Array], Dict[str, jax.Array], Dict[str, jax.Array], Dict[str, jax.Array]]:
    """Empirically estimates each searched field's starting value, scale,
    and valid range from `n_probe` samples of its own distribution - more
    robust than relying on a distribution's `.mean()`/`.stddev()`, which
    silently gets the wrong answer for a distribution like `examples/
    hparam_search.py`'s `CategoricalUniform` (its `.sample()` maps a
    Categorical's sampled index through a domain list, but doesn't
    override `.mean()`/`.stddev()` to match).

    Args:
        hparams_distr (Dict[str, distrax.Distribution]): One distribution
            per searched field.
        n_probe (int): Samples drawn per field.
        key (jax.Array): PRNG key, split once per field.

    Returns:
        Tuple[Dict, Dict, Dict, Dict]: `(theta, scale, lo, hi)` - each
        field's probe mean, probe std (floored to avoid a degenerate
        zero-sigma field), and probe min/max. `lo`/`hi` bound every
        candidate `sample_antithetic_candidates` produces and every ES
        update step - without them, nothing stops the search's random
        walk drifting a field outside the range `hparams_distr` was
        ever meant to describe (confirmed in practice: an unclipped
        search drifted `gae_lambda` - only ever valid in `[0, 1]` - to
        `1.005`, and separately drifted a `learning_rate` down to
        exactly `0.0` and got stuck there, since a learning rate of
        zero produces no fitness signal to climb back out with).
    """
    theta: Dict[str, jax.Array] = {}
    scale: Dict[str, jax.Array] = {}
    lo: Dict[str, jax.Array] = {}
    hi: Dict[str, jax.Array] = {}
    for k, distr in hparams_distr.items():
        key, sample_key = jax.random.split(key)
        raw_samples = distr.sample(seed=sample_key, sample_shape=(n_probe,))
        samples = jnp.asarray(raw_samples, dtype=jnp.float32)
        theta[k] = jnp.mean(samples)
        scale[k] = jnp.maximum(jnp.std(samples), 1e-8)
        lo[k] = jnp.min(samples)
        hi[k] = jnp.max(samples)
    return theta, scale, lo, hi


def sample_antithetic_candidates(
    theta: Dict[str, jax.Array],
    scale: Dict[str, jax.Array],
    lo: Dict[str, jax.Array],
    hi: Dict[str, jax.Array],
    pop_size: int,
    sigma: float,
    key: jax.Array,
) -> Tuple[Dict[str, jax.Array], Dict[str, jax.Array]]:
    """One ES generation's population: `pop_size // 2` i.i.d. standard-
    normal noise vectors per field, mirrored (antithetic sampling) to
    fill the rest of the population, then `theta + sigma * scale *
    noise` per field, clipped to `[lo[k], hi[k]]` (see
    `probe_hparam_field_stats`).

    Args:
        theta (Dict[str, Array]): Current per-field mean.
        scale (Dict[str, Array]): Per-field noise scale (see
            `probe_hparam_field_stats`).
        lo (Dict[str, Array]): Per-field lower bound.
        hi (Dict[str, Array]): Per-field upper bound.
        pop_size (int): Population size. Must be even.
        sigma (float): Noise scale, in units of `scale`.
        key (jax.Array): PRNG key, split once per field.

    Returns:
        Tuple[Dict, Dict]: `(noise, candidates)`, each `Dict[str,
        Array]` shaped `(pop_size,)` per field - `noise` is what the ES
        gradient estimate is computed from, `candidates` is what
        actually gets trained.
    """
    half = pop_size // 2
    noise: Dict[str, jax.Array] = {}
    candidates: Dict[str, jax.Array] = {}
    for k in theta:
        key, noise_key = jax.random.split(key)
        eps = jax.random.normal(noise_key, (half,))
        noise[k] = jnp.concatenate([eps, -eps])
        values = theta[k] + sigma * scale[k] * noise[k]
        candidates[k] = jnp.clip(values, lo[k], hi[k])
    return noise, candidates
