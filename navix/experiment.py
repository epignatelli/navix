from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace, fields
import time
import warnings
from typing import Dict, Optional, Tuple

import distrax
import jax
import jax.numpy as jnp
import wandb
import wandb.util
from navix.agents.agent import Agent
from navix.environments.environment import Environment


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

            def log_seed(seed):
                config = {**vars(self), **asdict(self.agent.hparams)}
                config.update(seed=seed)
                run = wandb.init(project=self.name, config=config, group=self.group)
                print("Logging results for seed:", seed)
                log = jax.tree.map(lambda x: x[seed], logs)
                self.agent.log_to_wandb_on_train_end(log, run=run)
                run.finish()

            # each seed's wandb.init -> log -> finish cycle is
            # independent, network-I/O-bound work - run them
            # concurrently so wall-clock time doesn't scale with the
            # number of seeds. wandb.init() itself being called
            # concurrently from multiple threads is unverified beyond
            # wandb's own multiprocessing-oriented docs - if this ever
            # shows up as misattributed/corrupted runs rather than an
            # exception, that's the first place to look.
            with ThreadPoolExecutor(max_workers=max(len(self.seeds), 1)) as executor:
                list(executor.map(log_seed, self.seeds))

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

        def log_hparam_set(i):
            print("Logging results for hparam set:", search_set)
            hparams = jax.tree.map(lambda x: x[i], search_set)
            config = {**vars(self), **asdict(hparams)}
            run = wandb.init(project=self.name, config=config, group=self.group)
            # average over seeds
            log = jax.tree.map(lambda x: jnp.mean(x[i], axis=0), logs)
            self.agent.log_to_wandb_on_train_end(log, run=run)
            run.finish()

        with ThreadPoolExecutor(max_workers=max(len_search_set, 1)) as executor:
            list(executor.map(log_hparam_set, range(len_search_set)))

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
