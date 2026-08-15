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
            # Masked mean via sum/count instead of boolean-indexing
            # (`lengths[mask]`) - mask[mask] has a *dynamic* shape (however
            # many episodes finished this update, which varies every call),
            # so JAX must recompile a fresh XLA program for every distinct
            # count it hasn't seen before. Profiling showed this was ~60%
            # of total per-call logging time. jnp.where + jnp.sum keeps the
            # shape fixed at (T, N) -> scalar regardless of how many
            # entries are masked, so XLA compiles it once and reuses it.
            mask_count = jnp.sum(mask)
            # log episode length
            if "lengths" in logs:
                lengths: jax.Array = logs.pop("lengths")  # (T, N)
                logs["perf/episode_length"] = (
                    jnp.sum(jnp.where(mask, lengths, 0)) / mask_count
                )
                msg += f", Length: {logs['perf/episode_length']}"

            # log returns
            if "returns" in logs:
                returns = logs.pop("returns")  # (T, N)
                logs["perf/returns"] = jnp.sum(jnp.where(mask, returns, 0)) / mask_count
                logs["perf/success_rate"] = (
                    jnp.sum(jnp.where(mask, returns == 1.0, False)) / mask_count
                )
                msg += f", Returns: {logs['perf/returns']}, Success Rate: {logs['perf/success_rate']}"

        msg += f", Logging time cost: {time.time() - start_time}"
        wandb.log(logs, step=step)

    def log_on_train_end(self, logs):
        print(jax.tree.map(lambda x: x.shape, logs))
        len_logs = len(logs["iter/updates"])
        updates = logs["iter/updates"]
        for step in range(len_logs):
            # skip steps self.log() would discard anyway (see the
            # log_frequency check below), before paying for the
            # device-to-host transfer of indexing into every array in
            # `logs` - wandb.log() and this per-step tree indexing were
            # previously done unconditionally for every recorded step,
            # which is why disabling wandb logging entirely (do_log=False)
            # was so much faster than leaving it on
            if updates[step] % self.hparams.log_frequency != 0:
                continue
            step_logs = {k: jax.tree.map(lambda x: x[step], v) for k, v in logs.items()}
            self.log(step_logs)
