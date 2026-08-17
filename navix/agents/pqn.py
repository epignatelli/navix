# Copyright 2023 The Navix Authors.

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# This implementation of PQN follows:
#   Gallici et al., "Simplifying Deep Temporal Difference Learning"
#   https://arxiv.org/abs/2407.04811
# with the reference single-file implementation at
#   https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/pqn.py
# as the ground truth for exact algorithmic details (target computation,
# exploration schedule, network init) - the paper's prose alone
# underdetermines these. Structurally this file mirrors `navix.agents.ppo`
# (rollout collection -> flatten -> minibatch SGD, same TrainingState/
# Buffer/Agent conventions), since the two share that training-loop shape;
# the algorithm itself is a different family (off-policy-free Q-learning,
# not an on-policy actor-critic).
"""PQN ("Parallelised Q-Network"): an online, parallel-environment deep
Q-learning agent with no replay buffer and no target network.

Standard DQN needs both a replay buffer (to decorrelate updates from a
single, highly autocorrelated trajectory) and a target network (to keep
the regression target from chasing itself as the online network
updates) to avoid diverging. PQN's core claim is that neither is
necessary once the Q-network is regularized with LayerNorm and trained
on data from many parallel environments at once (as `PPO` already does
for the same decorrelation reason): the online network's own current
weights can serve as the bootstrap target, and LayerNorm keeps that
self-referential regression stable instead of blowing up. What
survives is close to the simplest thing that could be called deep
Q-learning: collect a rollout, regress Q(s, a) towards a bootstrapped
return, repeat.

Per-update loop (`update`):
  1. `collect_experience` - `hparams.num_steps` steps across
     `hparams.num_envs` parallel envs, epsilon-greedy over the online
     network's own Q-values (`distrax.EpsilonGreedy`, annealed via
     `epsilon`). Each transition caches `max_a Q(s_t, a)` (`Buffer.value`)
     at collection time - the reference implementation's `values` array -
     so the target computation below never needs to re-run the network
     over already-visited states.
  2. `evaluate_experience` - a Q(lambda) return over the whole rollout,
     mixing the one-step bootstrap (`(1 - q_lambda)`, using the cached
     `value` one step ahead) with the multi-step return
     (`q_lambda`, chaining through the *target*, not the cached value, at
     one step ahead) - literally `rlax.lambda_returns`, whose own
     docstring calls out this exact use: "Q(lambda): v_t = max(q_t,
     axis=-1)". Computed once per rollout, not re-evaluated per epoch
     the way `PPO.update` re-evaluates its GAE targets - the reference
     implementation runs `update_epochs` of SGD against one fixed set of
     targets, since there's no importance-ratio correction here that
     would make re-evaluating them mid-epoch meaningful.
  3. `hparams.num_epochs` passes of shuffled-minibatch MSE regression
     (`q_loss`) of `Q(s_t, a_t)` towards that fixed target.

No target network: the bootstrap value at every step - both the cached
`Buffer.value` and the final bootstrap in `evaluate_experience` - comes
from the same `train_state.params` the rollout was collected with, not
a separately-updated copy. No replay buffer: every minibatch this
update draws on is a shuffled slice of the rollout `collect_experience`
just produced, used once, then discarded - unlike DQN, nothing here
persists across `update` calls.

On `PQNHparams`' defaults for gridworld tasks: `num_epochs`/
`num_minibatches`/`exploration_fraction`/`end_e` are set higher/lower
than CleanRL's CartPole-tuned reference script, not because that script
is wrong, but because a `budget`-frame run buys many fewer *rollouts*
here (`num_updates = budget // (num_steps * num_envs)`) than CartPole's
own `total_timesteps` example uses, and PQN gets no benefit from the
extra fixed-target minibatch passes a replay-buffer method would - the
only way to extract more learning signal per rollout is more epochs
over it. Verified empirically (not just reasoned about): the external
`rejax` package's own PQN, with its per-environment-tuned config for
`Navix-Empty-6x6-v0` (`num_epochs=8`, `num_minibatches=128`,
`exploration_fraction=0.3`, `end_e=0.1`, `gamma=0.9`), reaches ~100%
success where this file's original CleanRL-derived defaults reached
~72% at 1M frames on the (comparably simple) `Navix-Empty-5x5-v0` -
and dropping rejax's exact config into *this* implementation
reproduces its ~100% result, confirming the target computation itself
was never the problem. `rejax`'s own target computation, on inspection
(`rejax/algos/pqn.py`), turned out to diverge from the official
reference it's adapted from (Gallici's own
`mttga/purejaxql/purejaxql/pqn_gymnax.py`): the official version caches
`Q` at the state a transition *starts* from and shifts it forward by
one step when bootstrapping (`this module's Buffer.value` does the
same); `rejax`'s adaptation caches `Q` at the state the transition
*lands in* instead, which is the wrong operand one step early in the
backward recursion. This module's `evaluate_experience` follows the
official convention, not rejax's.
"""
import time
from typing import Dict, Tuple

import distrax
import jax
import jax.numpy as jnp
from jax import Array
import optax
from flax.training.train_state import TrainState
from flax import struct
from flax.linen import FrozenDict as Params
import rlax

from navix.observations import rgb
from navix.agents.agent import Agent, HParams
from navix.environments.environment import Timestep
from navix.states import State

from .models import QNetwork


class PQNHparams(HParams):
    budget: int = struct.field(pytree_node=False, default=1_000_000)
    """Number of environment frames to train for."""
    num_envs: int = struct.field(pytree_node=False, default=16)
    """Number of parallel environments to run."""
    num_steps: int = struct.field(pytree_node=False, default=128)
    """Number of steps to run in each environment per update."""
    num_minibatches: int = struct.field(pytree_node=False, default=32)
    """Number of minibatches to split the rollout into. Higher than
    CleanRL's CartPole-tuned default (8) - see this module's docstring
    on gridworld-appropriate defaults for why."""
    num_epochs: int = struct.field(pytree_node=False, default=8)
    """Number of shuffled-minibatch passes per update over the rollout's
    (fixed, not re-evaluated) Q(lambda) targets - "update_epochs" in the
    reference implementation. Higher than CleanRL's CartPole-tuned
    default (4) - see this module's docstring."""
    q_lambda: float = 0.65
    """Mixing parameter for the Q(lambda) return target - see
    `rlax.lambda_returns`."""
    lr: float = 2.5e-4
    """Starting learning rate."""
    anneal_lr: bool = struct.field(pytree_node=False, default=True)
    """Whether to anneal the learning rate linearly to 0 at the end of training."""
    max_grad_norm: float = 10.0
    """Maximum norm for gradient clipping."""
    start_e: float = 1.0
    """Initial epsilon for epsilon-greedy exploration."""
    end_e: float = 0.1
    """Final epsilon for epsilon-greedy exploration. Higher than
    CleanRL's CartPole-tuned default (0.05) - see this module's
    docstring."""
    exploration_fraction: float = 0.3
    """Fraction of `budget` over which epsilon anneals from `start_e` to
    `end_e`; held at `end_e` for the remainder. Shorter than CleanRL's
    CartPole-tuned default (0.5) - see this module's docstring."""
    hidden_size: int = 64
    """Hidden layer size of the Q-network."""


class Buffer(struct.PyTreeNode):
    done: jax.Array
    action: jax.Array
    reward: jax.Array
    value: jax.Array
    obs: jax.Array
    info: Dict[str, jax.Array]
    t: jax.Array
    state: State


class TrainingState(TrainState):
    env_state: Timestep
    rng: jax.Array
    frames: jax.Array
    epoch: jax.Array


class PQN(Agent):
    hparams: PQNHparams
    network: QNetwork = struct.field(pytree_node=False)

    def epsilon(self, frames: Array) -> Array:
        """Linear anneal from `start_e` to `end_e` over
        `exploration_fraction * budget` frames, then held at `end_e`."""
        hp = self.hparams
        duration = hp.exploration_fraction * hp.budget
        slope = (hp.end_e - hp.start_e) / duration
        return jnp.maximum(slope * frames + hp.start_e, hp.end_e)

    def collect_experience(
        self, train_state: TrainingState
    ) -> Tuple[TrainingState, Buffer]:
        def _env_step(
            collection_state: Tuple[Timestep, jax.Array, jax.Array], _
        ) -> Tuple[Tuple[Timestep, jax.Array, jax.Array], Buffer]:
            env_state, rng, frames = collection_state
            frames = frames + self.hparams.num_envs

            # SELECT ACTION: epsilon-greedy over the online network's own
            # Q-values - no target network, so this is the same
            # `train_state.params` the previous update just trained.
            # `distrax.EpsilonGreedy`, not `rlax.epsilon_greedy` - rlax's
            # own docstring flags that one as pending deprecation in
            # favor of this.
            rng, _rng = jax.random.split(rng)
            q_values = jnp.asarray(
                train_state.apply_fn(train_state.params, env_state.observation)
            )
            action = jnp.asarray(
                distrax.EpsilonGreedy(q_values, self.epsilon(frames)).sample(
                    seed=_rng
                )
            )
            value = jnp.max(q_values, axis=-1)  # cached max_a Q(o_t, a)

            # STEP ENV
            new_env_state = jax.vmap(self.env.step, in_axes=(0, 0))(env_state, action)
            transition = Buffer(
                done=new_env_state.is_done(),  # done(o_{t+1})
                action=action,  # a_t
                reward=new_env_state.reward,  # R(o_t, a_t)
                value=value,  # max_a Q(o_t, a)
                obs=env_state.observation,  # o_t
                info=new_env_state.info,  # info(o_{t+1})
                t=env_state.t,  # t
                state=env_state.state,  # s_t
            )
            return (new_env_state, rng, frames), transition

        (env_state, rng, frames), experience = jax.lax.scan(
            _env_step,
            (train_state.env_state, train_state.rng, train_state.frames),
            None,
            self.hparams.num_steps,
        )
        train_state = train_state.replace(env_state=env_state, rng=rng, frames=frames)
        return train_state, experience

    def evaluate_experience(
        self, train_state: TrainingState, experience: Buffer
    ) -> jax.Array:
        """The Q(lambda) return target, computed once per rollout (not
        re-evaluated per epoch - see this module's docstring). `values`
        are the *cached* `max_a Q(o_t, a)` from collection time
        (`Buffer.value`); the one bootstrap not already cached is
        `max_a Q(o_T, a)` at the post-rollout observation, under the
        same (not-yet-updated-this-round) params."""
        last_q = jnp.asarray(
            train_state.apply_fn(
                train_state.params, train_state.env_state.observation
            )
        )
        last_val = jnp.max(last_q, axis=-1)
        next_values = jnp.concatenate(
            [experience.value[1:], last_val[None]], axis=0
        )  # max_a Q(o_{t+1}, a), (T, N)
        discount = self.env.gamma * (1.0 - experience.done)  # (T, N)
        returns = jax.vmap(
            rlax.lambda_returns, in_axes=(1, 1, 1, None), out_axes=1
        )(experience.reward, discount, next_values, self.hparams.q_lambda)
        return jnp.asarray(returns)

    def q_loss(
        self,
        params: Params,
        transition_batch: Buffer,
        targets: Array,
    ) -> Tuple[Array, Dict]:
        # already vmapped over the minibatch
        q_values = jax.vmap(self.network.apply, in_axes=(None, 0))(
            params, transition_batch.obs
        )
        q_taken = rlax.batched_index(q_values, transition_batch.action.astype(jnp.int32))
        # rlax.l2_loss carries the conventional 1/2 factor (its own
        # docstring notes this follows Bishop's PRML, not the
        # unscaled squared error some other texts call "L2 loss") - a
        # constant rescaling of the gradient `lr` already absorbs.
        loss = jnp.mean(rlax.l2_loss(q_taken, targets))
        logs = {
            "loss/q_loss": loss,
            "agent/q_value": q_taken.mean(),
            "agent/target": targets.mean(),
        }
        return loss, logs

    def sgd_step(
        self,
        train_state: TrainingState,
        minibatch: Tuple[Buffer, jax.Array],
    ) -> Tuple[TrainingState, Dict]:
        traj_batch, targets = minibatch
        grad_fn = jax.value_and_grad(self.q_loss, has_aux=True)
        (_, logs), grads = grad_fn(train_state.params, traj_batch, targets)
        train_state = train_state.apply_gradients(grads=grads)
        return train_state, logs

    def update(self, train_state: TrainingState, _) -> Tuple[TrainingState, Dict]:
        minibatch_size = (
            self.hparams.num_envs
            * self.hparams.num_steps
            // self.hparams.num_minibatches
        )
        # collect experience and compute its Q(lambda) targets ONCE - see
        # this module's docstring for why, unlike PPO, this isn't
        # re-evaluated inside the epoch loop below.
        train_state, experience = self.collect_experience(train_state)
        targets = self.evaluate_experience(train_state, experience)

        n_samples = minibatch_size * self.hparams.num_minibatches
        assert (
            n_samples == self.hparams.num_steps * self.hparams.num_envs
        ), "batch size must be equal to number of steps * number of envs"

        rng = train_state.rng
        for _ in range(self.hparams.num_epochs):
            rng, rng_1 = jax.random.split(rng)
            permutation = jax.random.permutation(rng_1, n_samples)
            samples = (experience, targets)  # (T, N, ...)
            samples = jax.tree.map(
                lambda x: x.reshape((n_samples,) + x.shape[2:]), samples
            )  # (T * N, ...)
            shuffled_batch = jax.tree.map(
                lambda x: jnp.take(x, permutation, axis=0), samples
            )  # (T * N, ...)

            minibatches = jax.tree.map(
                lambda x: jnp.reshape(
                    x, (self.hparams.num_minibatches, -1) + tuple(x.shape[1:])
                ),
                shuffled_batch,
            )
            train_state, logs = jax.lax.scan(self.sgd_step, train_state, minibatches)

        train_state = train_state.replace(
            rng=rng,
            epoch=train_state.epoch + self.hparams.num_epochs,
        )
        logs = jax.tree.map(lambda x: jnp.mean(x), logs)

        learning_rate = train_state.opt_state[1].hyperparams["learning_rate"]  # type: ignore

        logs["done_mask"] = experience.done
        logs["returns"] = experience.info["return"]
        logs["lengths"] = experience.t

        logs["iter/frames"] = train_state.frames
        logs["iter/epochs"] = train_state.epoch
        logs["iter/updates"] = train_state.step
        logs["iter/learning_rate"] = learning_rate
        logs["agent/epsilon"] = self.epsilon(train_state.frames)

        if self.hparams.log_render:
            b = jax.random.randint(rng, (), 0, self.hparams.num_envs)
            logs["render/human"] = jax.vmap(rgb)(
                jax.tree.map(lambda x: x[:, b], experience.state)
            ).transpose(
                (0, 3, 1, 2)
            )  # (T, 3, H, W)

        if self.hparams.debug:
            jax.debug.callback(self.log_to_wandb, logs, experience)

        return train_state, logs

    def train(self, rng: jax.Array) -> Tuple[TrainingState, Dict]:
        # INIT NETWORK
        rng, _rng = jax.random.split(rng)
        init_x = self.env.observation_space.sample(_rng)
        params = self.network.init(_rng, init_x)

        num_updates = self.hparams.budget // (
            self.hparams.num_steps * self.hparams.num_envs
        )

        def linear_schedule(count):
            frac = (
                1.0
                - (count // (self.hparams.num_minibatches * self.hparams.num_epochs))
                / num_updates
            )
            return self.hparams.lr * frac

        lr = linear_schedule if self.hparams.anneal_lr else self.hparams.lr
        # RAdam, not Adam - matches the reference implementation; PQN's
        # stability claims were made with this optimizer, not verified
        # against plain Adam.
        tx = optax.chain(
            optax.clip_by_global_norm(self.hparams.max_grad_norm),
            optax.inject_hyperparams(optax.radam)(learning_rate=lr),
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, self.hparams.num_envs)
        env_state = jax.vmap(self.env.reset)(reset_rng)

        train_state = TrainingState.create(
            apply_fn=jax.vmap(self.network.apply, in_axes=(None, 0)),
            params=params,
            tx=tx,
            env_state=env_state,
            rng=rng,
            frames=jnp.asarray(0, dtype=jnp.int32),
            epoch=jnp.asarray(0, dtype=jnp.int32),
        )
        start_time = time.time()
        train_state, logs = jax.lax.scan(self.update, train_state, length=num_updates)
        elapsed = time.time() - start_time
        logs["iter/fps"] = jnp.asarray([train_state.frames / elapsed] * num_updates)
        logs["iter/wall_time"] = jnp.asarray([elapsed] * num_updates)

        return train_state, logs
