from dataclasses import dataclass
import time
import warnings
from typing import Dict, Tuple

import numpy as np
import wandb
import jax
import jax.numpy as jnp
from flax import struct
from flax.training.train_state import TrainState

from ..environments.environment import Environment


def masked_mean(values: jax.Array, mask: jax.Array, axis=None) -> jax.Array:
    """Mean of `values` over entries where `mask` is True, computed via a masked
    sum/count rather than boolean-indexing `values[mask]`. Boolean-indexing
    produces a dynamically-shaped result (its size depends on how many
    entries are True, which usually varies across calls), forcing JAX to
    recompile a fresh XLA program for every distinct count it hasn't seen
    before. This keeps the output shape fixed - `values.shape` reduced
    over `axis` - regardless of how many entries are masked, so XLA
    compiles it once and reuses it.

    Args:
        values (Array): The values to average.
        mask (Array): A boolean array, broadcastable to `values.shape`,
            selecting which entries to include.
        axis: The axis or axes to reduce over. `None` reduces to a scalar.

    Returns:
        Array: The mean of `values` where `mask` is True, reduced over `axis`."""
    mask = jnp.asarray(mask, dtype=jnp.bool_)
    return jnp.sum(jnp.where(mask, values, 0), axis=axis) / jnp.sum(mask, axis=axis)


_REQUIRED_LOG_KEYS: Dict[str, str] = {
    "done_mask": "which steps ended an episode",
    "lengths": "per-step episode length",
    "returns": "per-step episodic return",
}


def derive_episodic_metrics(logs: Dict[str, jax.Array]) -> Dict[str, jax.Array]:
    """Reduces the raw per-step buffers (`done_mask`, `lengths`,
    `returns`) that `Agent.train` returns into `perf/episode_length`,
    `perf/returns`, `perf/success_rate` - one point per training
    update, masked-mean over completed episodes only.

    `Agent.log_to_wandb` computes the same values, but one training
    update at a time (for live wandb logging); this is the batched
    equivalent, for reducing an entire already-finished `logs` history
    in one call - used by `Experiment.run_hparam_search` (as the ES
    search's fitness signal) and `navix.benchmarks.plotting` (as
    `plot_metric`/`plot_dashboard`'s `perf/*` inputs).

    Args:
        logs (Dict[str, Array]): The `logs` pytree returned by
            `Experiment.run()`, `Experiment.run_hparam_search()`, or a
            bare `Agent.train()` call. Must contain `done_mask`/
            `lengths`/`returns`, shaped `(..., num_steps, num_envs)` -
            any number of leading batch dimensions (e.g. seeds,
            hparam sets) is supported.

    Returns:
        Dict[str, Array]: `logs`, plus `perf/episode_length`,
        `perf/returns` and `perf/success_rate` (shape: `logs`'
        leading batch dimensions, with `num_steps` and `num_envs`
        reduced away). `logs` itself is not mutated.

    Raises:
        KeyError: If `logs` is missing `done_mask`, `lengths`, or
            `returns`.
    """
    missing = [key for key in _REQUIRED_LOG_KEYS if key not in logs]
    if missing:
        reasons = ", ".join(f"{key!r} ({_REQUIRED_LOG_KEYS[key]})" for key in missing)
        raise KeyError(f"logs is missing required key(s): {reasons}.")

    metrics = dict(logs)
    mask = jnp.asarray(logs["done_mask"], dtype=jnp.bool_)
    metrics["perf/episode_length"] = masked_mean(logs["lengths"], mask, axis=(-2, -1))
    returns = logs["returns"]
    metrics["perf/returns"] = masked_mean(returns, mask, axis=(-2, -1))
    metrics["perf/success_rate"] = masked_mean(returns == 1.0, mask, axis=(-2, -1))
    return metrics


class HParams(struct.PyTreeNode):
    debug: bool = struct.field(pytree_node=False, default=False)
    """Whether to run in debug mode."""
    log_frequency: int = struct.field(pytree_node=False, default=1)
    """How often to log results."""
    log_render: bool = struct.field(pytree_node=False, default=False)


class Agent(struct.PyTreeNode):
    """Two strategies exist for looking at a run's results, and they trade off
    against each other:

    - `Experiment.run(log_to_wandb=True)` (the default) streams metrics to
      Weights & Biases as training progresses, via `log_to_wandb`/
      `log_to_wandb_on_train_end` below. This is the slow path - real
      network I/O, roughly linear in the number of seeds - but gives you
      wandb's hosted dashboards, run comparison, etc.
    - `Experiment.run(log_to_wandb=False)` skips wandb entirely and just
      returns `logs` (the same pytree these methods consume) directly - no
      network calls, so it's dramatically faster. Pair it with
      `navix.benchmarks.plotting` to get a local matplotlib dashboard from
      `logs` instead of a wandb one. See issue #60.
    """

    hparams: HParams
    env: Environment
    """On the base class since every agent needs one - unlike
    algorithm-specific internals (e.g. PPO's `sgd_step`)."""

    def train(self, rng: jax.Array) -> Tuple[TrainState, Dict[str, jax.Array]]:
        raise NotImplementedError

    def log_to_wandb(self, logs, inspectable=None, run=None):
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
            mask = jnp.asarray(logs.pop("done_mask"), dtype=jnp.bool_)  # (T, N)
            # log episode length
            if "lengths" in logs:
                lengths: jax.Array = logs.pop("lengths")  # (T, N)
                logs["perf/episode_length"] = masked_mean(lengths, mask)
                msg += f", Length: {logs['perf/episode_length']}"

            # log returns
            if "returns" in logs:
                returns = logs.pop("returns")  # (T, N)
                logs["perf/returns"] = masked_mean(returns, mask)
                logs["perf/success_rate"] = masked_mean(returns == 1.0, mask)
                msg += f", Returns: {logs['perf/returns']}, Success Rate: {logs['perf/success_rate']}"

        msg += f", Logging time cost: {time.time() - start_time}"
        # Use the explicit Run object when given, rather than the
        # module-level wandb.log, which only tracks one implicit
        # "current run" - lets a caller log to a specific Run explicitly
        # (e.g. Experiment.run's per-seed loop) rather than relying on
        # whichever run wandb.init() last set as "current".
        (run or wandb).log(logs, step=step)

    def log_to_wandb_on_train_end(self, logs, run=None):
        print(jax.tree.map(lambda x: x.shape, logs))
        len_logs = len(logs["iter/updates"])
        updates = logs["iter/updates"]
        for step in range(len_logs):
            if updates[step] % self.hparams.log_frequency != 0:
                continue
            step_logs = {k: jax.tree.map(lambda x: x[step], v) for k, v in logs.items()}
            self.log_to_wandb(step_logs, run=run)

    def log(self, logs, inspectable=None, run=None):
        """Deprecated: use `log_to_wandb` instead."""
        warnings.warn(
            "Agent.log is deprecated, use Agent.log_to_wandb instead - this "
            "method only ever sent data to wandb, so the name now says so.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.log_to_wandb(logs, inspectable=inspectable, run=run)

    def log_on_train_end(self, logs, run=None):
        """Deprecated: use `log_to_wandb_on_train_end` instead."""
        warnings.warn(
            "Agent.log_on_train_end is deprecated, use "
            "Agent.log_to_wandb_on_train_end instead - this method only "
            "ever sent data to wandb, so the name now says so.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.log_to_wandb_on_train_end(logs, run=run)
