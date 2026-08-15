from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace, fields
import multiprocessing
import time
import warnings
from typing import Dict, Optional, Tuple

import distrax
import numpy as np
import jax
import jax.numpy as jnp
import wandb
import wandb.util
from navix.agents.agent import Agent, HParams
from navix.environments.environment import Environment


def _to_numpy(x):
    """Converts a JAX array (or anything array-like) to a plain numpy array,
    leaving everything else untouched. Used to strip out live JAX device
    buffers before crossing a process boundary - see `_log_run_to_wandb`."""
    return np.asarray(x) if isinstance(x, jax.Array) else x


def _cpu_only_worker_init():
    """`ProcessPoolExecutor` initializer (runs once per worker process,
    before any task). Hides the GPU from the child process entirely, so no
    worker can ever create a CUDA context - belt-and-braces on top of using
    `spawn` (see `_log_run_to_wandb`'s docstring for why `spawn` alone is
    already safe here)."""
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"


def _log_run_to_wandb(
    hparams: HParams, project: str, config: dict, group: str, log_np: dict
):
    """Runs one wandb Run (`init` -> `log_to_wandb_on_train_end` -> `finish`)
    for a single seed/hparam-set, in its own OS process rather than a
    thread.

    wandb's SDK carries global/singleton state that isn't actually isolated
    per `Run` instance across threads - concurrent `wandb.init()` calls from
    multiple threads have been observed to race into a real login attempt
    even under `wandb offline`, and even serializing just `wandb.init()`
    still left runs getting corrupted ("Run (...) is finished. The call to
    `log` will be ignored.") when multiple threads logged to their own
    `Run` objects concurrently. Separate OS processes don't share that
    state at all, matching wandb's own multiprocessing-oriented docs.

    Safe from the usual JAX/CUDA fork hazard (forking a process after CUDA
    is initialized in the parent is undefined behavior) because the caller
    uses `multiprocessing.get_context("spawn")`: `spawn` starts a fresh
    interpreter with nothing inherited from the parent, rather than forking
    its memory (including its live CUDA context) - so this is safe
    regardless of whether the parent's training has already finished, not
    because of the timing. `_cpu_only_worker_init` additionally hides the
    GPU from this process entirely, so nothing here can create a CUDA
    context even by accident.

    `log_np` must already be plain numpy (no JAX arrays) - see `_to_numpy` -
    since JAX device buffers aren't guaranteed to survive being pickled
    across a process boundary.
    """
    import wandb

    run = wandb.init(project=project, config=config, group=group)
    Agent(hparams=hparams).log_to_wandb_on_train_end(log_np, run=run)
    run.finish()


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
        """Default function to run the experiment. This function compiles the training function, trains the agent, and logs the results.

        Two strategies exist for looking at the results, and they trade off
        against each other:

        - `log_to_wandb=True` (the default) streams metrics to Weights &
          Biases as training progresses. This is the slow path - real
          network I/O, roughly linear in the number of seeds.
        - `log_to_wandb=False` skips wandb entirely; `logs` (this
          method's second return value) is returned either way, so with
          wandb off you get it back much faster, with no network calls at
          all. Pair it with `navix.plotting` to get a local matplotlib
          dashboard from `logs` instead of a wandb one. See issue #60.

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
        train_state, logs = train_fn(rng)
        training_time = time.time() - start_time
        print(f"Training time cost: {training_time}")

        if not self.agent.hparams.debug and log_to_wandb:
            print("Logging final results to wandb...")
            start_time = time.time()

            hparams_np = jax.tree.map(_to_numpy, self.agent.hparams)

            # each seed's wandb.init -> log -> finish cycle is independent,
            # network-I/O-bound work - run them concurrently, in separate
            # processes (not threads - see _log_run_to_wandb's docstring),
            # so wall-clock time doesn't scale with the number of seeds.
            ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=max(len(self.seeds), 1),
                mp_context=ctx,
                initializer=_cpu_only_worker_init,
            ) as executor:
                futures = []
                for seed in self.seeds:
                    config = {
                        "name": self.name,
                        "agent": str(self.agent),
                        "env": str(self.env),
                        "env_id": self.env_id,
                        "group": self.group,
                        "seed": seed,
                        **asdict(hparams_np),
                    }
                    log_np = jax.tree.map(lambda x, s=seed: _to_numpy(x[s]), logs)
                    print("Logging results for seed:", seed)
                    futures.append(
                        executor.submit(
                            _log_run_to_wandb,
                            hparams_np,
                            self.name,
                            config,
                            self.group,
                            log_np,
                        )
                    )
                for f in futures:
                    f.result()

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
        self, hparams_distr: Dict[str, distrax.Distribution], pop_size: int
    ):
        """Function to run a hyperparameter search for the experiment. This function \
        samples hyperparameters from the given distributions, trains the agent, and \
        logs the results.
        
        Args:
            hparams_distr (Dict[str, distrax.Distribution]): A dictionary of \
            hyperparameter distributions. The keys are the hyperparameter names, which \
            must exist in `self.agent.hparams`, and the values are the corresponding \
            distributions.
            pop_size (int): The number of hyperparameter sets to sample.

        Returns:
            Tuple: A tuple containing the final training states and the logs, batched \
            over the hyperparameter sets.
        """
        hparams_fields = fields(self.agent.hparams)
        for k in hparams_distr:
            member = list(filter(lambda x: x.name == k, hparams_fields))
            if (
                len(member) > 0
                and "pytree_node" in member[0].metadata
                and member[0].metadata["pytree_node"] == False
            ):
                raise ValueError(
                    f"Hyperparameter {k} is not a traceable pytree node. "
                    + f"Set pytree_node=True for {k} to include it into the hparam search."
                )

        search_set = []
        for seed in range(pop_size):
            hparams = self.agent.hparams
            key = jax.random.PRNGKey(seed)
            for k, distr in hparams_distr.items():
                hparams = replace(hparams, **{k: distr.sample(seed=key)})
            print("Hparams:", hparams)
            search_set.append(hparams)
        # transpose search set
        len_search_set = len(search_set)
        search_set = jax.tree.map(lambda *x: jnp.stack(x), *search_set)

        rngs = jnp.asarray([jax.random.PRNGKey(seed) for seed in self.seeds])

        def search(hparam_set_sample):
            agent = self.agent.replace(hparams=hparam_set_sample)
            return jax.vmap(agent.train)(rngs)

        print("Running hyperparameter search with the following configuration:")
        print(search_set)

        print("Compiling search function...")
        start_time = time.time()
        search_fn = jax.jit(jax.vmap(search)).lower(search_set).compile()
        compilation_time = time.time() - start_time
        print(f"Compilation time cost: {compilation_time}")

        print("Searching for optimal hyperparameters...")
        start_time = time.time()
        train_states, logs = search_fn(search_set)
        search_time = time.time() - start_time
        print(f"Search time cost: {search_time}")

        print("Logging final results to wandb...")
        start_time = time.time()

        # see run()'s logging block for why this is multiprocessing (with
        # spawn + a CPU-only worker init), not threading.
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=max(len_search_set, 1),
            mp_context=ctx,
            initializer=_cpu_only_worker_init,
        ) as executor:
            futures = []
            for i in range(len_search_set):
                hparams_i_np = jax.tree.map(lambda x: _to_numpy(x[i]), search_set)
                config = {
                    "name": self.name,
                    "env_id": self.env_id,
                    "group": self.group,
                    **asdict(hparams_i_np),
                }
                # average over seeds
                log_np = jax.tree.map(
                    lambda x, i=i: _to_numpy(jnp.mean(x[i], axis=0)), logs
                )
                print("Logging results for hparam set:", hparams_i_np)
                futures.append(
                    executor.submit(
                        _log_run_to_wandb,
                        hparams_i_np,
                        self.name,
                        config,
                        self.group,
                        log_np,
                    )
                )
            for f in futures:
                f.result()

        logging_time = time.time() - start_time

        print("Hyperparameter search complete")
        total_time = 0
        print(f"Compilation time cost: {compilation_time}")
        total_time += compilation_time
        print(f"Search time cost: {search_time}")
        total_time += search_time
        print(f"Logging time cost: {logging_time}")
        total_time += logging_time
        print(f"Total time cost: {total_time}")
        return train_states, logs
