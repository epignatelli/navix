from dataclasses import dataclass
import time
from typing import Dict, Tuple

import numpy as np
import wandb
import jax
import jax.numpy as jnp
from flax import struct
from flax.training.train_state import TrainState


class HParams(struct.PyTreeNode):
    debug: bool = struct.field(pytree_node=False, default=False)
    """Whether to run in debug mode."""
    log_frequency: int = struct.field(pytree_node=False, default=1)
    """How often to log results."""
    log_render: bool = struct.field(pytree_node=False, default=False)


class Agent(struct.PyTreeNode):
    hparams: HParams

    def train(self, rng: jax.Array) -> Tuple[TrainState, Dict[str, jax.Array]]:
        raise NotImplementedError

    def log(self, logs, inspectable=None):
        if len(logs) == 0 or logs["iter/updates"] % self.hparams.log_frequency != 0:
            return

        start_time = time.time()
        msg = f"Update Step: {logs['iter/updates']}, Frames: {logs['iter/frames']}"
        step = jnp.asarray(logs["iter/updates"], dtype=jnp.int32)

        # log renders
        if self.hparams.log_render:
            render_human = logs.pop("render/human")  # (T, 3, H, W)
            logs[f"render/human"] = wandb.Video(np.array(render_human), fps=4)

        if "done_mask" in logs:
            mask = jnp.asarray(logs.pop("done_mask"), dtype=jnp.bool)  # (T, N)
            # log episode length
            if "lengths" in logs:
                lengths: jax.Array = logs.pop("lengths")  # (T, N)
                episode_lengths = lengths[mask]  # (K,)
                logs["perf/episode_length"] = jnp.mean(episode_lengths)
                msg += f", Length: {logs['perf/episode_length']}"

            # log returns
            if "returns" in logs:
                returns = logs.pop("returns")  # (T, N)
                final_returns = returns[mask]  # (K,)
                logs["perf/returns"] = jnp.mean(final_returns)
                logs["perf/success_rate"] = jnp.mean(final_returns == 1.0)
                msg += f", Returns: {logs['perf/returns']}, Success Rate: {logs['perf/success_rate']}"

        msg += f", Logging time cost: {time.time() - start_time}"
        wandb.log(logs, step=step)

    def log_on_train_end(self, logs):
        print(jax.tree.map(lambda x: x.shape, logs))
        len_logs = len(logs["iter/updates"])
        for step in range(len_logs):
            step_logs = {k: jax.tree.map(lambda x: x[step], v) for k, v in logs.items()}
            self.log(step_logs)
