from dataclasses import asdict
import time
import jax
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
        seed: int,
        debug: bool = False,
    ):
        self.name = name
        self.budget = budget
        self.agent = agent
        self.env = env
        self.seed = seed
        self.debug = debug

    def run(self):
        config = {**vars(self), **asdict(self.agent.hparams)}
        wandb.init(project=self.name, config=config)

        rng = jax.random.PRNGKey(self.seed)

        print("Compiling training function...")
        start_time = time.time()
        train_fn = jax.jit(self.agent.train).lower(rng).compile()
        compilation_time = time.time() - start_time
        print(f"Compilation time cost: {compilation_time}")

        print("Training agent...")
        start_time = time.time()
        train_state, logs = train_fn(rng)
        training_time = time.time() - start_time
        print(f"Training time cost: {training_time}")

        if not self.debug:
            print("Logging final results to wandb...")
            start_time = time.time()
            self.agent.log_on_train_end(logs)
            wandb.log({})
            logging_time = time.time() - start_time
            print(f"Logging time cost: {logging_time}")

        print("Training complete")
        print(f"Compilation time cost: {compilation_time}")
        print(f"Training time cost: {training_time}")
        total_time = compilation_time + training_time
        if not self.debug:
            print(f"Logging time cost: {logging_time}")
            total_time += logging_time
        print(f"Total time cost: {total_time}")
        return train_state, logs
