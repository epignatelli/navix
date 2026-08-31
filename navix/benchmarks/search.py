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

"""Evolution-Strategies hyperparameter search for `AlgorithmEntry`
submissions - the same algorithm `navix.experiment.Experiment.
run_hparam_search` uses (see `navix.es`), but operating on any
`TrainingCurve`-returning trainable rather than a navix `Agent`'s own
`HParams` specifically. This is what makes it usable for an external
library's entry (e.g. rejax) that has no navix `Agent`/`HParams` at
all - `search_hparams` only ever calls `trainable(hparams, rng)` and
reads back a `TrainingCurve`, never touching whatever's inside
`trainable` itself.

Deliberately NOT wired into `Benchmark`/`AlgorithmEntry` - `Benchmark`
is algorithm-agnostic by design (`entry.train` is opaque to it; an
entry "doesn't have to build a navix Agent at all", see this package's
`__init__.py`), and hyperparameters aren't a structured, exposed field
on `AlgorithmEntry` at all. Searching them is inherently entry-specific
(what fields exist, how they're threaded into `train`), so it stays a
tool an entry's own `run.py` opts into, not a `Benchmark.run_env` flag -
see benchmarks/README.md for how a `run.py` uses this."""
from typing import Callable, Dict, Optional, Tuple

import distrax
import jax
import jax.numpy as jnp
import optax

from .benchmark import TrainingCurve
from ..es import probe_hparam_field_stats, sample_antithetic_candidates

Trainable = Callable[[Dict[str, jax.Array], jax.Array], TrainingCurve]
"""`(hparams, rng) -> TrainingCurve` - `env_id`/`budget` (and anything
else `entry.train` needs) are expected to already be fixed via closure,
e.g. `lambda hparams, rng: entry_train_fn(hparams, env_id, budget, rng)`.
`hparams` overrides only the fields named in `search_hparams`'s own
`hparams_distr` - everything else is whatever the closure's own
defaults are."""


def search_hparams(
    trainable: Trainable,
    hparams_distr: Dict[str, distrax.Distribution],
    seeds: Tuple[int, ...],
    pop_size: int = 8,
    num_generations: int = 10,
    sigma: float = 1.0,
    solver: Optional[optax.GradientTransformation] = None,
    n_probe: int = 256,
) -> Tuple[Dict[str, float], jax.Array]:
    """Evolution-strategies hyperparameter search (see `navix.es` for
    the shared antithetic-sampling/probe-statistics math, and
    `Experiment.run_hparam_search`'s docstring for the full algorithm
    description - this is the same algorithm, generalized past navix's
    own `Agent`/`HParams`).

    Each generation: sample an antithetic population of `pop_size`
    hyperparameter sets around the current mean, call `trainable` on
    every one of them (vmapped over both the population and `seeds`),
    score each by its last-20%-mean `episodic_returns` (`TrainingCurve.
    last_percent_mean`, averaged over seeds), then take an ES step. The
    best-scoring hyperparameter set actually evaluated across every
    generation - not the (never directly trained) mean trajectory
    itself - is what's returned.

    Args:
        trainable (Trainable): `(hparams, rng) -> TrainingCurve` - see
            `Trainable`'s own docstring for the closure convention.
        hparams_distr (Dict[str, distrax.Distribution]): One
            distribution per searched field - seeds that field's
            starting value/scale/valid range (via `navix.es.
            probe_hparam_field_stats`), not resampled every generation.
            Every candidate and every ES update is clipped to that
            field's empirical `[min, max]` - without it, nothing stops
            the search drifting a field outside the range this
            distribution was ever meant to describe (e.g. a `gae_lambda`
            distribution meant to express "search within `[0.8, 0.99]`"
            wouldn't stop the search drifting past `1.0`, an invalid
            value).
        seeds (Tuple[int, ...]): PRNG seeds `trainable` is vmapped over
            per candidate - must have more than one, so fitness isn't
            just RNG luck for a single rollout.
        pop_size (int): Population size per generation. Must be even -
            antithetic sampling pairs (+/-).
        num_generations (int): Number of ES update steps.
        sigma (float): Noise scale, in units of each field's own
            empirical probe std.
        solver (optax.GradientTransformation, optional): The ES mean
            update rule. Defaults to `optax.sgd(0.1)`.
        n_probe (int): Samples drawn from each field's distribution to
            estimate that field's starting value and scale.

    Returns:
        Tuple[Dict[str, float], Array]: The best-scoring hyperparameter
        set actually evaluated across every generation (plain Python
        floats, ready to splice into whatever config `trainable`'s
        closure builds from), and its fitness (last-20%-mean
        `episodic_returns`, averaged over `seeds`).

    Raises:
        ValueError: If `seeds` has one or fewer entries, or `pop_size`
            is odd.
    """
    if len(seeds) <= 1:
        raise ValueError(f"seeds must have more than one entry, got {seeds!r}.")
    if pop_size % 2 != 0:
        raise ValueError(f"pop_size must be even (antithetic sampling pairs +/-), got {pop_size}.")
    if solver is None:
        solver = optax.sgd(0.1)

    theta, scale, lo, hi = probe_hparam_field_stats(hparams_distr, n_probe, jax.random.PRNGKey(0))
    opt_state = solver.init(theta)

    rngs = jnp.asarray([jax.random.PRNGKey(seed) for seed in seeds])

    def population_train(hparams_batch: Dict[str, jax.Array]) -> TrainingCurve:
        def one_member(hp: Dict[str, jax.Array]) -> TrainingCurve:
            return jax.vmap(lambda rng: trainable(hp, rng))(rngs)

        return jax.vmap(one_member)(hparams_batch)

    # A regular jax.jit call (not .lower().compile()) so the compiled
    # program is cached and reused across every generation below -
    # hparams_batch's pytree structure/shapes never change generation
    # to generation, only its leaf values do.
    search_fn = jax.jit(population_train)

    print("Running evolution-strategies hyperparameter search:")
    print(f"  fields: {list(hparams_distr.keys())}, pop_size: {pop_size}, num_generations: {num_generations}")
    print(f"  starting point: {theta}")

    best_hparams: Optional[Dict[str, float]] = None
    best_fitness = -jnp.inf

    for generation in range(num_generations):
        gen_key = jax.random.PRNGKey(generation)
        noise, candidates = sample_antithetic_candidates(theta, scale, lo, hi, pop_size, sigma, gen_key)

        curves = jax.block_until_ready(search_fn(candidates))
        fitness = jnp.mean(curves.last_percent_mean().episodic_returns, axis=-1)  # (pop_size,)

        shaped = (fitness - jnp.mean(fitness)) / jnp.sqrt(jnp.var(fitness) + 1e-8)
        grad = {k: jnp.mean(shaped * noise[k]) for k in candidates}
        # optax solvers descend a loss's gradient - feed -grad (the
        # gradient of -fitness) so solver.update's output ascends
        # fitness instead, then apply_updates-style addition (not
        # subtraction) matching optax's own convention.
        updates, opt_state = solver.update({k: -g for k, g in grad.items()}, opt_state, theta)
        theta = {k: jnp.clip(v + scale[k] * updates[k], lo[k], hi[k]) for k, v in theta.items()}

        gen_best_idx = int(jnp.argmax(fitness))
        gen_best_fitness = float(fitness[gen_best_idx])
        print(
            f"Generation {generation}: fitness best={gen_best_fitness:.4f} "
            f"mean={float(jnp.mean(fitness)):.4f} worst={float(jnp.min(fitness)):.4f}"
        )

        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_hparams = {k: float(v[gen_best_idx]) for k, v in candidates.items()}

    print(f"Best hparams found: {best_hparams}")
    print(f"Best fitness found: {best_fitness}")

    assert best_hparams is not None
    return best_hparams, jnp.asarray(best_fitness)
