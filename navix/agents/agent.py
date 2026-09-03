"""`Agent`: the common interface `Experiment` trains, and `HParams`: the
base its per-algorithm hyperparameter structs extend.

The concrete agents - `PPO`, `PQN`, `Dreamer` - each subclass `Agent`,
implement `train`, and carry their own `HParams` subclass. This module
also holds the shared logging helpers (`masked_mean`,
`derive_episodic_metrics`) and the `agent/train/*` metric contract every
agent's `train` must return.
"""

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


REQUIRED_LOG_KEYS: Dict[str, str] = {
    "agent/train/done_mask": "which steps ended an episode",
    "agent/train/lengths": "per-step episode length",
    "agent/train/returns": "per-step episodic return",
}
"""`agent/train/*` is `Agent.train`'s own guaranteed-floor namespace -
see its docstring for exactly what's verified common across every
navix agent and why (as opposed to `agent/diagnostics/*`, algorithm-
specific and never guaranteed)."""


def derive_episodic_metrics(logs: Dict[str, jax.Array]) -> Dict[str, jax.Array]:
    """Reduces the raw per-step buffers (`agent/train/done_mask`,
    `agent/train/lengths`, `agent/train/returns`) that `Agent.train`
    returns into `agent/episode/length`, `agent/episode/returns`,
    `agent/episode/success_rate` - one point per training update,
    masked-mean over completed episodes only. Not itself a per-episode
    log (nothing in `logs` is - see `Agent.train`'s docstring): a
    single aggregate statistic over however many episodes happened to
    complete that update.

    `Agent.log_to_wandb` computes the same values, but one training
    update at a time (for live wandb logging); this is the batched
    equivalent, for reducing an entire already-finished `logs` history
    in one call - used by `Experiment.run_hparam_search` (as the ES
    search's fitness signal) and `navix.benchmarks.plotting` (as
    `plot_metric`/`plot_dashboard`'s `agent/episode/*` inputs).

    Args:
        logs (Dict[str, Array]): The `logs` pytree returned by
            `Experiment.run()`, `Experiment.run_hparam_search()`, or a
            bare `Agent.train()` call. Must contain
            `agent/train/done_mask`/`agent/train/lengths`/
            `agent/train/returns`, shaped `(..., num_steps, num_envs)`
            - any number of leading batch dimensions (e.g. seeds,
            hparam sets) is supported.

    Returns:
        Dict[str, Array]: `logs`, plus `agent/episode/length`,
        `agent/episode/returns` and `agent/episode/success_rate`
        (shape: `logs`' leading batch dimensions, with `num_steps` and
        `num_envs` reduced away). `logs` itself is not mutated.

    Raises:
        KeyError: If `logs` is missing `agent/train/done_mask`,
            `agent/train/lengths`, or `agent/train/returns`.
    """
    missing = [key for key in REQUIRED_LOG_KEYS if key not in logs]
    if missing:
        reasons = ", ".join(f"{key!r} ({REQUIRED_LOG_KEYS[key]})" for key in missing)
        raise KeyError(f"logs is missing required key(s): {reasons}.")

    metrics = dict(logs)
    mask = jnp.asarray(logs["agent/train/done_mask"], dtype=jnp.bool_)
    metrics["agent/episode/length"] = masked_mean(logs["agent/train/lengths"], mask, axis=(-2, -1))
    returns = logs["agent/train/returns"]
    metrics["agent/episode/returns"] = masked_mean(returns, mask, axis=(-2, -1))
    metrics["agent/episode/success_rate"] = masked_mean(returns == 1.0, mask, axis=(-2, -1))
    return metrics


class HParams(struct.PyTreeNode):
    """Base for every agent's hyperparameter struct (`PPOHparams`,
    `PQNHparams`, `DreamerHparams`). Holds only the fields common to all;
    each subclass adds its own (learning rate, rollout length, ...).
    Frozen - use `.replace(...)` for a modified copy (this is what the
    hyperparameter search does)."""

    debug: bool = struct.field(pytree_node=False, default=False)
    """If `True`, agents run extra `jax.debug` callbacks and per-step
    wandb logging. Slow; off by default."""
    log_frequency: int = struct.field(pytree_node=False, default=1)
    """Log to wandb every `log_frequency` training updates (`1` = every
    update)."""
    log_render: bool = struct.field(pytree_node=False, default=False)
    """If `True`, agents also emit an `rgb` rollout video under
    `render/human` in their logs."""


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
        """Trains this agent from scratch and returns `logs`: the
        training history every downstream consumer (`log_to_wandb`,
        `Experiment`, `navix.benchmarks`) reads.

        Every navix namespace that ends up in a `logs`-shaped dict is
        prefixed by *who produced it*, not just what kind of thing it
        is - `agent/*` here, `experiment/*` for what `Experiment` adds
        after `train()` already returned, `benchmark/*` for what
        `Benchmark.summary`/`details`/`diagnostics.npz` add on top of
        that (see those modules' own docstrings). Within `agent/*`,
        `logs`' keys split into exactly two namespaces - what's
        structurally guaranteed (`agent/train/*`, from the shared
        collect/derive path every concrete agent already goes through,
        not something each implementation writes by hand) and what's
        genuinely bespoke (`agent/diagnostics/*`) - verified directly
        against every navix agent's own `update`/`train`, not assumed:

        - **`agent/train/*`, guaranteed**: `agent/train/done_mask`/
          `agent/train/returns`/`agent/train/lengths` (the raw,
          per-step interaction stream - required by
          `derive_episodic_metrics`, which raises `KeyError` if any is
          missing) and `agent/train/frames`/`agent/train/updates`
          (identical across every navix agent). None of these are
          themselves per-episode values - `returns`/`lengths` are
          dense running sums reset on episode boundary, only
          meaningful where `done_mask` is true (see `navix.
          environments.environment.Environment.step`'s
          `info["return"]` accumulation).
        - **`agent/diagnostics/*`, bespoke**: everything else,
          including things that look like they should be structural
          but aren't actually uniform - e.g. an epoch count or
          learning-rate schedule state exists for PPO/PQN but not in
          the same shape for Dreamer (three optimizers, not one) -
          alongside the obviously algorithm-specific values (PPO's
          `agent/diagnostics/entropy`/`agent/diagnostics/value_loss`/
          ...; PQN's `agent/diagnostics/q_loss`/`agent/diagnostics/
          epsilon`; Dreamer's `agent/diagnostics/model/*`/`agent/
          diagnostics/actor/*`/`agent/diagnostics/critic/*`). One
          shared prefix, no shared key names - a caller (e.g.
          `benchmarks/*/*/run.py`'s `TrainingCurve.diagnostics`
          construction) can always filter on
          `key.startswith("agent/diagnostics/")` without needing to
          know which specific keys a given agent happens to log.

        Neither of `train()`'s own two namespaces has anything called
        `agent/episode/*` or `experiment/costs/*`. `agent/episode/*` is
        `derive_episodic_metrics`'s own output, computed downstream
        from `agent/train/*`'s raw stream. `experiment/costs/wall_time`/
        `experiment/costs/fps` are added by `Experiment.run`/
        `run_hparam_search` after `train()` already returned - real
        wall-clock timing can't be measured from inside the `jax.jit`
        trace `train()` runs in. A bare `agent.train(rng)` call, not
        wrapped by `Experiment`, returns `logs` with neither.

        Args:
            rng (jax.Array): PRNG key for the whole training run.

        Returns:
            Tuple[TrainState, Dict[str, Array]]: The final train
            state, and `logs` as described above."""
        raise NotImplementedError

    def log_to_wandb(self, logs, inspectable=None, run=None):
        if len(logs) == 0 or logs["agent/train/updates"] % self.hparams.log_frequency != 0:
            return

        start_time = time.time()
        msg = f"Update Step: {logs['agent/train/updates']}, Frames: {logs['agent/train/frames']}"
        step = jnp.asarray(logs["agent/train/updates"], dtype=jnp.int32)

        # log renders
        if self.hparams.log_render:
            render_human = logs.pop("render/human")  # (T, 3, H, W)
            logs[f"render/human"] = wandb.Video(np.array(render_human), fps=4)

        if "agent/train/done_mask" in logs:
            mask = jnp.asarray(logs.pop("agent/train/done_mask"), dtype=jnp.bool_)  # (T, N)
            # log episode length
            if "agent/train/lengths" in logs:
                lengths: jax.Array = logs.pop("agent/train/lengths")  # (T, N)
                logs["agent/episode/length"] = masked_mean(lengths, mask)
                msg += f", Length: {logs['agent/episode/length']}"

            # log returns
            if "agent/train/returns" in logs:
                returns = logs.pop("agent/train/returns")  # (T, N)
                logs["agent/episode/returns"] = masked_mean(returns, mask)
                logs["agent/episode/success_rate"] = masked_mean(returns == 1.0, mask)
                msg += f", Returns: {logs['agent/episode/returns']}, Success Rate: {logs['agent/episode/success_rate']}"

        msg += f", Logging time cost: {time.time() - start_time}"
        # Use the explicit Run object when given, rather than the
        # module-level wandb.log, which only tracks one implicit
        # "current run" - lets a caller log to a specific Run explicitly
        # (e.g. Experiment.run's per-seed loop) rather than relying on
        # whichever run wandb.init() last set as "current".
        (run or wandb).log(logs, step=step)

    def log_to_wandb_on_train_end(self, logs, run=None):
        print(jax.tree.map(lambda x: x.shape, logs))
        len_logs = len(logs["agent/train/updates"])
        updates = logs["agent/train/updates"]
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
