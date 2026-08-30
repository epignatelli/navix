"""`Experiment`: trains one `Agent` against one `Environment`, across
`seeds`, optionally logging to wandb (`run`), or searches its
hyperparameters via Evolution Strategies (`run_hparam_search`)."""
from dataclasses import asdict, replace, fields
import time
import warnings
from typing import Dict, Optional, Tuple

import distrax
import jax
import jax.numpy as jnp
import optax
import wandb
import wandb.util
from navix.agents.agent import Agent, HParams
from navix.benchmarks.plotting import derive_episodic_metrics
from navix.environments.environment import Environment
from navix.es import probe_hparam_field_stats, sample_antithetic_candidates

# Logging to wandb is sequential, one seed/hparam-set at a time - both
# concurrency options that were tried here turned out not to hold up:
#
# - Threads: wandb's SDK carries global/singleton state that isn't
#   actually isolated per Run instance across threads. Concurrent
#   wandb.init() calls from multiple threads raced into a real login
#   attempt even under `wandb offline`; serializing just wandb.init()
#   with a lock still left runs corrupted ("Run (...) is finished. The
#   call to `log` will be ignored.") when multiple threads logged to
#   their own Run objects concurrently.
# - Processes (spawn, to stay safe w.r.t. JAX/CUDA - forking a process
#   after CUDA is initialized in the parent is undefined behavior
#   regardless of timing, since every thread but the calling one just
#   vanishes in the child, taking whatever locks it held with it):
#   measured on a real GPU across 1/2/4/8 seeds, spawning a process per
#   seed was 1.4-2.8x *slower* than sequential up to 4 seeds (each
#   worker re-imports the entire jax/flax/distrax/tensorflow_probability/
#   wandb stack from a cold interpreter), only broke even around 8, and
#   at 16 concurrent workers the pool outright crashed from resource
#   exhaustion (MemoryError, `ptxas` launch failures). Not worth the
#   complexity for a win that only shows up at seed counts nobody's
#   actually running by default.
#
# `run()`'s docstring below still calls this out explicitly, since it's
# a real cost worth knowing about before choosing `log_to_wandb=True`.


def _build_search_set(
    base_hparams: HParams, candidates: Dict[str, jax.Array], pop_size: int
) -> HParams:
    """Batches `pop_size` individually-`replace`'d copies of
    `base_hparams` (one per population member's `candidates`) into a
    single `HParams` pytree with a new leading population axis - ready
    for `jax.vmap`.

    Args:
        base_hparams (HParams): Every other (non-searched) field's value.
        candidates (Dict[str, Array]): This generation's per-field
            candidate values, each shaped `(pop_size,)` (see
            `navix.es.sample_antithetic_candidates`).
        pop_size (int): Population size.

    Returns:
        HParams: Every searched field's leaves shaped `(pop_size, ...)`.
    """
    search_set_list = []
    for i in range(pop_size):
        hparams_i = base_hparams
        for k, values in candidates.items():
            hparams_i = replace(hparams_i, **{k: values[i]})
        search_set_list.append(hparams_i)
    return jax.tree.map(lambda *x: jnp.stack(x), *search_set_list)


def _hparam_search_fitness(logs: Dict[str, jax.Array]) -> jax.Array:
    """One scalar fitness per population member, from `logs` (as
    returned by a `run_hparam_search` generation's `search_fn` call):
    last-20%-mean `perf/returns` (`navix.benchmarks.plotting.
    derive_episodic_metrics`), averaged over the seed axis.

    Args:
        logs (Dict[str, Array]): Shape `(pop_size, num_seeds,
            num_updates, num_steps, num_envs)` for `done_mask`/
            `returns`/`lengths`.

    Returns:
        Array: Shape `(pop_size,)`.
    """
    returns = derive_episodic_metrics(logs)["perf/returns"]  # (pop_size, num_seeds, num_updates)
    tail = max(1, int(returns.shape[-1] * 0.2))
    return jnp.mean(jnp.mean(returns[..., -tail:], axis=-1), axis=-1)


class Experiment:
    """A class to run an experiment with a given agent and environment.

    Args:
        name (str): The name of the experiment.
        agent (Agent): The agent to use in the experiment.
        env (Environment): The environment to use in the experiment.
        env_id (str): The ID of the environment.
        seeds (Tuple[int, ...]): The seeds to use in the experiment.
        group (str): The group to use in the experiment.

    Attributes:
        name (str): The name of the experiment.
        agent (Agent): The agent to use in the experiment.
        env (Environment): The environment to use in the experiment.
        env_id (str): The ID of the environment.
        seeds (Tuple[int, ...]): The seeds to use in the experiment.
        group (str): The group to use in the experiment.

    """

    def __init__(
        self,
        name: str,
        agent: Agent,
        env: Environment,
        env_id: str = "",
        seeds: Tuple[int, ...] = (0,),
        group: str = "",
    ):
        self.name = name
        self.agent = agent
        self.env = env
        self.env_id = env_id
        self.seeds = seeds
        self.group = group

    def run(self, log_to_wandb: bool = True, do_log: Optional[bool] = None):
        """Default function to run the experiment. This function compiles
        the training function, trains the agent, and logs the results.

        Two strategies exist for looking at the results, and they trade off
        against each other:

        - `log_to_wandb=True` (the default) streams metrics to Weights &
          Biases as training progresses. This is the slow path - real
          network I/O, roughly linear in the number of seeds.
        - `log_to_wandb=False` skips wandb entirely; `logs` (this
          method's second return value) is returned either way, so with
          wandb off you get it back much faster, with no network calls at
          all. Pair it with `navix.benchmarks.plotting` to get a local
          matplotlib dashboard from `logs` instead of a wandb one. See
          issue #60.

        Args:
            log_to_wandb (bool): Whether to log the results to wandb.
            do_log (bool, optional): Deprecated alias for `log_to_wandb`.

        Returns:
            Tuple: A tuple containing the final training state and the logs.
        """
        if do_log is not None:
            warnings.warn(
                "Experiment.run's `do_log` is deprecated, use "
                "`log_to_wandb` instead - the old name didn't say what "
                "it was actually turning on/off.",
                DeprecationWarning,
                stacklevel=2,
            )
            log_to_wandb = do_log

        print("Running experiment with the following configuration:")
        print(vars(self))
        rng = jnp.asarray([jax.random.PRNGKey(seed) for seed in self.seeds])

        print("Compiling training function...")
        start_time = time.time()
        train_fn = jax.jit(jax.vmap(self.agent.train)).lower(rng).compile()
        compilation_time = time.time() - start_time
        print(f"Compilation time cost: {compilation_time}")

        print("Training agent...")
        start_time = time.time()
        train_state, logs = jax.block_until_ready(train_fn(rng))
        training_time = time.time() - start_time
        print(f"Training time cost: {training_time}")

        # iter/fps and iter/wall_time can't be measured inside agent.train
        # itself - it runs inside the jax.jit trace just compiled above,
        # where time.time() only ever fires once, at trace-build time (see
        # each agent's train()). training_time above, timed here from
        # outside any trace and block_until_ready'd, is the real thing -
        # one number for the whole vmapped call (seeds train together, so
        # there's no per-seed breakdown), broadcast to match every other
        # per-seed/per-update logged key's shape. fps is derived from each
        # seed's own final iter/frames count (real, already correctly
        # accumulated by the scan) rather than hparams.budget, both because
        # not every Agent.hparams carries a budget field and because a
        # seed's actual frame count can land slightly under budget (budget
        # // (num_steps * num_envs) floors to a whole number of updates).
        num_updates = logs["iter/updates"].shape[-1]
        frames = jnp.mean(jnp.asarray(logs["iter/frames"])[..., -1])
        fps = frames / training_time
        logs["iter/wall_time"] = jnp.full((len(self.seeds), num_updates), training_time)
        logs["iter/fps"] = jnp.full((len(self.seeds), num_updates), fps)

        if not self.agent.hparams.debug and log_to_wandb:
            print("Logging final results to wandb...")
            start_time = time.time()

            for seed in self.seeds:
                config = {**vars(self), **asdict(self.agent.hparams)}
                config.update(seed=seed)
                run = wandb.init(project=self.name, config=config, group=self.group)
                print("Logging results for seed:", seed)
                log = jax.tree.map(lambda x: x[seed], logs)
                self.agent.log_to_wandb_on_train_end(log, run=run)
                run.finish()

            logging_time = time.time() - start_time
            print(f"Logging time cost: {logging_time}")

        print("Training complete")
        total_time = 0
        print(f"Compilation time cost: {compilation_time}")
        total_time += compilation_time
        print(f"Training time cost: {training_time}")
        total_time += training_time
        if not self.agent.hparams.debug and log_to_wandb:
            print(f"Logging time cost: {logging_time}")
            total_time += logging_time
        print(f"Total time cost: {total_time}")
        return train_state, logs

    def run_hparam_search(
        self,
        hparams_distr: Dict[str, distrax.Distribution],
        pop_size: int,
        num_generations: int = 10,
        sigma: float = 1.0,
        solver: Optional[optax.GradientTransformation] = None,
        n_probe: int = 256,
        log_to_wandb: bool = True,
    ) -> Tuple[HParams, jax.Array]:
        """Evolution-strategies hyperparameter search, adapted from
        OpenAI-ES (Salimans et al., 2017 - https://arxiv.org/abs/1703.03864)
        the way https://github.com/ESHyperscale/HyperscaleES's `open_es.py`
        applies it to neural network weights, here applied to a
        hyperparameter vector instead.

        Each generation: sample an antithetic (mirrored +/-) population of
        `pop_size` hyperparameter sets around the current mean, train all
        of them in one fused `jax.jit(jax.vmap(...))` call (the same shape
        `Experiment.run` itself uses), score each by its last-20%-mean
        `perf/returns` (`navix.benchmarks.plotting.derive_episodic_metrics`,
        averaged over `self.seeds`), then take an ES step: z-score the
        fitnesses, estimate a gradient from fitness-weighted noise, and
        update the mean via `solver`. The best-scoring hyperparameter set
        *actually evaluated* across every generation - not the (never
        directly trained) mean trajectory itself - is what's returned.

        Every searched field spans a different natural scale (`lr` ~1e-4,
        `gae_lambda` ~0.95, ...), so `sigma` is relative, not absolute: an
        empirical probe of `n_probe` samples from each field's own
        `hparams_distr` distribution gives that field's starting value
        (the probe's mean) and natural scale (the probe's std) - `sigma`
        is then how many of *those* per-field stds each generation's noise
        spans. This also sidesteps relying on a distribution's `.mean()`/
        `.stddev()`, which a distribution like `examples/hparam_search.py`'s
        `CategoricalUniform` (its `.sample()` maps a Categorical's sampled
        index through a domain list, but doesn't override `.mean()`/
        `.stddev()` to match) would silently get wrong.

        Args:
            hparams_distr (Dict[str, distrax.Distribution]): One
                distribution per searched field. Keys must name a field on
                `self.agent.hparams` with `pytree_node=True` (navix's
                continuous float hparams - `lr`, `clip_eps`, `gae_lambda`,
                ... - not `budget`/`num_envs`/etc., which stay
                structurally unsearchable). Each distribution seeds that
                field's starting value/scale (see above) - it does not
                keep resampling every generation.
            pop_size (int): Population size per generation. Must be even -
                each generation samples `pop_size // 2` noise vectors and
                mirrors them (antithetic sampling), the same
                variance-reduction trick `open_es.py` uses.
            num_generations (int): Number of ES update steps.
            sigma (float): Noise scale, in units of each field's own
                empirical probe std (see above).
            solver (optax.GradientTransformation, optional): The ES mean
                update rule. Defaults to `optax.sgd(0.1)`.
            n_probe (int): Samples drawn from each field's distribution to
                estimate that field's starting value and scale.
            log_to_wandb (bool): Whether to log per-generation fitness
                stats plus the final best candidate's training curve to
                wandb.

        Returns:
            Tuple[HParams, Array]: The best-scoring hyperparameter set
            actually evaluated across every generation, and its fitness
            (last-20%-mean `perf/returns`, averaged over `self.seeds`).

        Raises:
            ValueError: If `pop_size` is odd, or `hparams_distr` names a
                `pytree_node=False` field.
        """
        if pop_size % 2 != 0:
            raise ValueError(
                f"pop_size must be even (antithetic sampling pairs +/-), got {pop_size}."
            )

        hparams_fields = fields(self.agent.hparams)
        for k in hparams_distr:
            member = list(filter(lambda x: x.name == k, hparams_fields))
            if (
                len(member) > 0
                and "pytree_node" in member[0].metadata
                and member[0].metadata["pytree_node"] is False
            ):
                raise ValueError(
                    f"Hyperparameter {k} is not a traceable pytree node. "
                    + f"Set pytree_node=True for {k} to include it into the hparam search."
                )

        if solver is None:
            solver = optax.sgd(0.1)

        theta, scale, non_negative = probe_hparam_field_stats(
            hparams_distr, n_probe, jax.random.PRNGKey(0)
        )
        opt_state = solver.init(theta)

        rngs = jnp.asarray([jax.random.PRNGKey(seed) for seed in self.seeds])

        def search(hparam_set_sample):
            agent = self.agent.replace(hparams=hparam_set_sample)
            return jax.vmap(agent.train)(rngs)

        # A regular jax.jit call (not .lower().compile()) so the compiled
        # program is cached and reused across every generation below -
        # search_set's pytree structure/shapes never change generation to
        # generation, only its (searched) leaf values do.
        search_fn = jax.jit(jax.vmap(search))

        print("Running evolution-strategies hyperparameter search:")
        print(
            f"  fields: {list(hparams_distr.keys())}, pop_size: {pop_size}, "
            f"num_generations: {num_generations}"
        )
        print(f"  starting point: {theta}")
        print(f"  per-field scale: {scale}")

        if log_to_wandb:
            config = {
                **vars(self),
                "hparams_distr": list(hparams_distr.keys()),
                "pop_size": pop_size,
                "num_generations": num_generations,
                "sigma": sigma,
            }
            run = wandb.init(project=self.name, config=config, group=self.group)

        best_hparams: Optional[HParams] = None
        best_fitness = -jnp.inf
        best_logs: Optional[Dict] = None

        start_time = time.time()
        for generation in range(num_generations):
            gen_key = jax.random.PRNGKey(generation)
            noise, candidates = sample_antithetic_candidates(
                theta, scale, non_negative, pop_size, sigma, gen_key
            )
            search_set = _build_search_set(self.agent.hparams, candidates, pop_size)

            gen_start = time.time()
            _, logs = jax.block_until_ready(search_fn(search_set))
            gen_wall_time = time.time() - gen_start

            # Same reasoning as Experiment.run: iter/fps/iter/wall_time
            # can't be measured inside agent.train itself - fill them in
            # here instead, for whichever generation's logs end up
            # surfaced as best_logs below (plotting.py's MANDATORY_METRICS
            # contract expects both to be present).
            gen_num_updates = logs["iter/updates"].shape[-1]
            gen_frames = jnp.mean(jnp.asarray(logs["iter/frames"])[..., -1])
            gen_fps = gen_frames / gen_wall_time
            gen_logs_shape = (pop_size, len(self.seeds), gen_num_updates)
            logs["iter/wall_time"] = jnp.full(gen_logs_shape, gen_wall_time)
            logs["iter/fps"] = jnp.full(gen_logs_shape, gen_fps)

            fitness = _hparam_search_fitness(logs)  # (pop_size,)

            shaped = (fitness - jnp.mean(fitness)) / jnp.sqrt(jnp.var(fitness) + 1e-8)
            grad = {k: jnp.mean(shaped * noise[k]) for k in candidates}
            # optax solvers descend a loss's gradient - feed -grad (the
            # gradient of -fitness) so solver.update's output ascends
            # fitness instead, then apply_updates-style addition (not
            # subtraction) matching optax's own convention.
            updates, opt_state = solver.update({k: -g for k, g in grad.items()}, opt_state, theta)
            theta = {k: value + scale[k] * updates[k] for k, value in theta.items()}
            for k, value in theta.items():
                if non_negative[k]:
                    theta[k] = jnp.maximum(value, 0.0)

            gen_best_idx = int(jnp.argmax(fitness))
            gen_best_fitness = float(fitness[gen_best_idx])
            print(
                f"Generation {generation}: fitness best={gen_best_fitness:.4f} "
                f"mean={float(jnp.mean(fitness)):.4f} worst={float(jnp.min(fitness)):.4f}"
            )
            if log_to_wandb:
                run.log(
                    {
                        "fitness/best": gen_best_fitness,
                        "fitness/mean": float(jnp.mean(fitness)),
                        "fitness/worst": float(jnp.min(fitness)),
                    },
                    step=generation,
                )

            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_hparams = jax.tree.map(lambda x: x[gen_best_idx], search_set)
                best_logs = jax.tree.map(lambda x: jnp.mean(x[gen_best_idx], axis=0), logs)

        search_time = time.time() - start_time
        print(f"Search time cost: {search_time}")
        print(f"Best hparams found: {best_hparams}")
        print(f"Best fitness found: {best_fitness}")

        if log_to_wandb:
            assert best_hparams is not None and best_logs is not None
            self.agent.log_to_wandb_on_train_end(best_logs, run=run)
            run.finish()

        assert best_hparams is not None
        return best_hparams, jnp.asarray(best_fitness)
