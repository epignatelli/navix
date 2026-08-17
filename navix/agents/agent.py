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


class CostAnalysis(struct.PyTreeNode):
    flops: float
    """Roughly one training update's worth of floating-point
    operations, from `jax.jit(agent.train).lower(rng).compile()
    .cost_analysis()['flops']` - see `Agent.cost_analysis`'s docstring
    for what this compiled program actually contains and why its cost
    lands near "one update", not the whole training run."""
    memory_bytes: float
    """Peak device memory (bytes) for that same compiled program:
    `argument_size_in_bytes + temp_size_in_bytes + output_size_in_bytes`
    from `.compile().memory_analysis()` - a proxy for peak usage (real
    peak can differ slightly due to buffer reuse XLA performs), not an
    exact measurement."""
    compile_time_seconds: float
    """Wall-clock time to `.lower(rng).compile()` the same program -
    the one-time XLA compilation cost paid once per distinct input
    shape/dtype (see `Experiment.run`'s own "Compilation time cost"
    logging for the equivalent measurement on a real training run).
    Unlike `flops`/`memory_bytes`, hardware- and XLA-version-sensitive
    in the same way `iter/fps`/`iter/wall_time` are."""


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
      `navix.plotting` to get a local matplotlib dashboard from `logs`
      instead of a wandb one. See issue #60.
    """

    hparams: HParams
    env: Environment
    """Every concrete `Agent` (PPO/Dreamer/PQN, and any future one)
    trains against exactly one `Environment` - promoted onto this base
    class rather than redeclared per subclass, since it's a genuine part
    of what any RL agent fundamentally needs, not an internal
    implementation detail specific to how a particular agent trains
    (unlike, say, PPO's `sgd_step`, which is PPO-specific machinery, not
    something every agent has)."""

    def train(self, rng: jax.Array) -> Tuple[TrainState, Dict[str, jax.Array]]:
        raise NotImplementedError

    def cost_analysis(self, rng: jax.Array) -> CostAnalysis:
        """Estimates this agent's training compute (FLOPs), peak
        memory, and compile time - implemented once, here, in terms of
        `train`, the *only* method every `Agent` is guaranteed to have.
        Deliberately does not assume any concrete agent's internal
        structure (e.g. that it has a `sgd_step`/`update` method with a
        particular signature) - a per-agent override would need that
        kind of assumption, which breaks for any agent (including an
        external submission) that doesn't happen to share navix's own
        agents' shape, so there isn't one: this base-class
        implementation is the only implementation.

        What's actually measured: `jax.jit(self.train).lower(rng)
        .compile()`, then that compiled program's own
        `cost_analysis()`/`memory_analysis()`. This necessarily
        includes whatever environment interaction `train` performs
        internally - `train` is the only contract `Agent` makes, and
        nothing in that contract says where (or whether) an
        implementation's own env-touching and env-free computation are
        separable - so this reports *total* training compute, not an
        agent-only figure with environment cost subtracted out.

        It still lands close to "one update's cost", not the whole
        run's, for a structural reason rather than a deliberate
        choice: every navix-shipped agent's `train` is a one-time init
        followed by `jax.lax.scan(self.update, ..., length=num_updates)`,
        and XLA's `cost_analysis()` on a compiled `scan`/`while` loop
        reports the cost of *one* loop iteration, not `length` copies
        of it (verified empirically - `cost_analysis()['flops']` on a
        `scan` of `length` in `{1, 10, 100}` returns the same value all
        three times). An agent whose `train` isn't scan-shaped would
        get a different number here - e.g. one that unrolls its whole
        budget in Python would report the FULL run's cost - which is a
        real difference in what gets measured across agents, not a bug
        in this method; `train` being the only guaranteed hook means
        this can't be normalized away without assuming more structure
        than the `Agent` contract promises."""
        start = time.time()
        compiled = jax.jit(self.train).lower(rng).compile()
        compile_time_seconds = time.time() - start

        flops = compiled.cost_analysis().get("flops", float("nan"))
        mem = compiled.memory_analysis()
        memory_bytes = (
            mem.argument_size_in_bytes + mem.temp_size_in_bytes + mem.output_size_in_bytes
        )
        return CostAnalysis(
            flops=flops,
            memory_bytes=memory_bytes,
            compile_time_seconds=compile_time_seconds,
        )

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

    def log(self, logs, inspectable=None, run=None):
        """Deprecated: use `log_to_wandb` instead."""
        warnings.warn(
            "Agent.log is deprecated, use Agent.log_to_wandb instead - this "
            "method only ever sent data to wandb, so the name now says so.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.log_to_wandb(logs, inspectable=inspectable, run=run)

    def log_to_wandb_on_train_end(self, logs, run=None):
        print(jax.tree.map(lambda x: x.shape, logs))
        len_logs = len(logs["iter/updates"])
        updates = logs["iter/updates"]
        for step in range(len_logs):
            # skip steps log_to_wandb() would discard anyway (see the
            # log_frequency check below), before paying for the
            # device-to-host transfer of indexing into every array in
            # `logs` - wandb.log() and this per-step tree indexing were
            # previously done unconditionally for every recorded step,
            # which is why disabling wandb logging entirely
            # (log_to_wandb=False) was so much faster than leaving it on
            if updates[step] % self.hparams.log_frequency != 0:
                continue
            step_logs = {k: jax.tree.map(lambda x: x[step], v) for k, v in logs.items()}
            self.log_to_wandb(step_logs, run=run)

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
