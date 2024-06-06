from dataclasses import asdict
import time
from typing import Tuple
import jax
import jax.numpy as jnp
import wandb
from navix.agents.agent import Agent
from navix.environments.environment import Environment


class Experiment:
    def __init__(
        self,
        name: str,
        budget: int,
        agent: Agent,
        env: Environment,
        env_id: str = "",
        seeds: Tuple[int, ...] = (0,),
    ):
        self.name = name
        self.budget = budget
        self.agent = agent
        self.env = env
        self.env_id = env_id
        self.seeds = seeds

    def run(self):
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

        if not self.agent.hparams.debug:
            print("Logging final results to wandb...")
            start_time = time.time()
            for seed in self.seeds:
                config = {**vars(self), **asdict(self.agent.hparams)}
                config.update(seed=seed)
                wandb.init(project=self.name, config=config)
                print("Logging results for seed:", seed)
                log = jax.tree.map(lambda x: x[seed], logs)
                self.agent.log_on_train_end(log)
                wandb.finish()
            logging_time = time.time() - start_time
            print(f"Logging time cost: {logging_time}")

        print("Training complete")
        total_time = 0
        print(f"Compilation time cost: {compilation_time}")
        total_time += compilation_time
        print(f"Training time cost: {training_time}")
        total_time += training_time
        if not self.agent.hparams.debug:
            print(f"Logging time cost: {logging_time}")
            total_time += logging_time
        print(f"Total time cost: {total_time}")
        return train_state, logs
