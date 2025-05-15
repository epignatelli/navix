import json
import os
import time
from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from flax import serialization
from rejax import get_algo
from rejax.evaluate import evaluate as rejax_evaluate


class Logger:
    def __init__(self, folder, name, metadata, num_seeds, use_wandb):
        self.folder = folder
        self.name = f"{name}_{time.time()}"
        self.metadata = metadata
        self.last_time = None
        self._log = {}
        self.timer = None
        self.num_seeds = num_seeds  # needed to aggregate vmapped results for wandb logs
        self.use_wandb = use_wandb

        if not os.path.exists(folder):
            os.makedirs(folder)

        print(f"Logging to {os.path.join(folder, name)}.{{json,ckpt}}")

    def log_once(self, data):
        self.metadata = {**self.metadata, **data}
        self.write_log()
        if self.use_wandb:
            for k, v in data.items():
                wandb.run.summary[k] = v

    def log(self, data, step):
        step = step.item()  # jax cpu callback returns numpy array
        if step not in self._log.keys():
            self._log[step] = []

        self._log[step].append(data)
        self.last_time[step] = time.process_time()
        self._log_wandb(step)

    def reset_timer(self):
        self.timer = time.process_time()
        self.last_time = {}

    def _convert(self, x):
        if isinstance(x, (np.ndarray, jnp.ndarray)):
            return x.tolist()
        return x

    def _log_wandb(self, step):
        if not self.use_wandb:
            return

        if len(self._log[step]) < self.num_seeds:
            return

        log_data = pd.DataFrame(self._log[step]).map(self._convert).to_dict("list")
        log_data = {k: np.mean(v) for k, v in log_data.items()}
        wandb.log(
            {"time/process_time": self.last_time[step] - self.timer, **log_data}, step
        )

    def write_log(self):
        file = os.path.join(self.folder, f"{self.name}.json")
        log_data = []
        for step in self._log.keys():
            log_data.append(
                {
                    "time/process_time": self.last_time[step] - self.timer,
                    "time/step": step,
                    **pd.DataFrame(self._log[step]).map(self._convert).to_dict("list"),
                }
            )
        log_data.sort(key=lambda x: x["time/step"])

        with open(file, "w+") as f:
            data = {
                **self.metadata,
                **pd.DataFrame(log_data).to_dict("list"),
            }
            json.dump(data, f)

    def write_checkpoint(self, ckpt):
        file = os.path.join(self.folder, f"{self.name}.ckpt")
        with open(file, "wb+") as f:
            f.write(serialization.to_bytes(ckpt))

        if self.use_wandb:
            wandb.save(file)


def make_evaluate(logger, env, env_params, num_seeds=20):
    def evaluate(algo, ts, rng):
        act = algo.make_act(ts)
        lengths, returns = rejax_evaluate(act, rng, env, env_params, num_seeds)
        jax.experimental.io_callback(
            logger.log,
            None,
            # Take mean over evaluation seeds
            {
                "episode_length": lengths.mean(axis=0),
                "episode_length_std": lengths.std(axis=0),
                "episode_length_max": lengths.max(axis=0),
                "episode_length_min": lengths.min(axis=0),
                "return": returns.mean(axis=0),
                "return_std": returns.std(axis=0),
                "return_max": returns.max(axis=0),
                "return_min": returns.min(axis=0),
            },
            ts.global_step,
        )

    return evaluate


def main(args, config):
    # Initialize logging
    escaped_env = config["env"].replace("/", "_")
    log_name = f"{escaped_env}_{args.algorithm}_{args.num_seeds}_{args.global_seed}"
    metadata = {
        "environment": config["env"],
        "algorithm": args.algorithm,
        "num_seeds": args.num_seeds,
        "global_seed": args.global_seed,
        "config": config,
    }
    logger = Logger(args.log_dir, log_name, metadata, args.num_seeds, args.use_wandb)
    logger.write_log()
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=metadata,
            name=log_name,
        )

    # Prepare train function and config
    algo = get_algo(args.algorithm)
    algo = algo.create(**deepcopy(config))  # Bug in pinned rejax version overwrites cfg
    evaluate = make_evaluate(logger, algo.env, algo.env_params)
    algo = algo.replace(eval_callback=evaluate)

    key = jax.random.PRNGKey(args.global_seed)
    keys = jax.random.split(key, args.num_seeds)
    vmap_train = jax.jit(jax.vmap(algo.train))

    # Time compilation
    start = time.process_time()
    lowered = vmap_train.lower(keys)
    time_lower = time.process_time() - start
    compiled = lowered.compile()
    time_compile = time.process_time() - time_lower
    vmap_train = compiled

    logger.active = True
    logger.log_once({"time/lower": time_lower, "time/compile": time_compile})
    logger.write_log()

    # Train
    logger.reset_timer()
    train_state, _ = vmap_train(keys)
    logger.write_log()
    if args.save_all_checkpoints:
        logger.write_checkpoint(train_state)
    else:
        train_state = jax.tree.map(lambda x: x[0], train_state)
        logger.write_checkpoint(train_state)


if __name__ == "__main__":
    import argparse

    from yaml import CLoader as Loader
    from yaml import load

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/navix_empty_5x5_1m.yaml",
        help="Path to configuration file.",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=1,
        help="Number of seeds to use.",
    )
    parser.add_argument(
        "--save-all-checkpoints",
        action="store_true",
        help="Save checkpoints of all seeds.",
    )
    parser.add_argument(
        "--global-seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="results",
        help="Directory to store logs.",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Use wandb for logging.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="project",
        help="Wandb project name.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default="entity",
        help="Wandb entity name.",
    )
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = load(f, Loader=Loader)

    if args.use_wandb:
        import wandb

    main(args, config[args.algorithm])
