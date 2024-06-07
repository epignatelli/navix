# This implementation of PPO is broadly inspired by:
# https://github.com/luchris429/purejaxrl/blob/main/purejaxrl/ppo.py
# which is in turn inspired by:
# https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, Tuple

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
from navix.environments import Environment
from navix.environments.environment import Timestep
from navix.states import State

from .models import ActorCriticRNN, RNNState


@dataclass
class PPOHparams(HParams):
    budget: int = 1_000_000
    """Number of environment frames to train for."""
    num_envs: int = 16
    """Number of parallel environments to run."""
    num_steps: int = 128
    """Number of steps to run in each environment per update."""
    num_minibatches: int = 8
    """Number of minibatches to split the data into for training."""
    num_epochs: int = 1
    """Number of epochs to train for."""
    gae_lambda: float = 0.95
    """Lambda parameter of the TD(lambda) return."""
    clip_eps: float = 0.2
    """PPO clip parameter."""
    ent_coef: float = 0.01
    """Entropy coefficient in the total loss."""
    vf_coef: float = 0.5
    """Value function coefficient in the total loss."""
    max_grad_norm: float = 0.5
    """Maximum gradient norm for clipping."""
    lr: float = 2.5e-4
    """Starting learning rate."""
    anneal_lr: bool = True
    """Whether to anneal the learning rate linearly to 0 at the end of training."""
    debug: bool = False
    """Whether to run in debug mode."""
    log_render: bool = False
    """Whether to log environment renderings."""
    normalise_advantage: bool = True
    """Whether to normalise the advantages in the PPO loss."""
    clip_value_loss: bool = True
    """Whether to clip the value loss in the PPO loss."""
    log_frequency: int = 1
    """How often to log results."""


HiddenState = Tuple[RNNState, RNNState]


class Buffer(struct.PyTreeNode):
    carry: Tuple[RNNState, RNNState]
    done: jax.Array
    action: jax.Array
    reward: jax.Array
    log_prob: jax.Array
    obs: jax.Array
    info: Dict[str, jax.Array]
    t: jax.Array
    state: State


class TrainingState(TrainState):
    env_state: Timestep
    rng: jax.Array
    frames: jax.Array
    epoch: jax.Array
    carry: HiddenState
    policy: Callable[
        [Params, HiddenState, Array, Array], Tuple[HiddenState, distrax.Distribution]
    ] = struct.field(pytree_node=False)
    value_fn: Callable[
        [Params, HiddenState, Array, Array], Tuple[HiddenState, Array]
    ] = struct.field(pytree_node=False)


class PPORNN(Agent):
    hparams: PPOHparams = struct.field(pytree_node=False)
    network: ActorCriticRNN = struct.field(pytree_node=False)
    env: Environment

    def collect_experience(
        self, train_state: TrainingState
    ) -> Tuple[TrainingState, Buffer]:
        def _env_step(
            collection_state: Tuple[Timestep, jax.Array, HiddenState], _
        ) -> Tuple[Tuple[Timestep, jax.Array, HiddenState], Buffer]:
            env_state, rng, carry = collection_state
            # SELECT ACTION
            rng, _rng = jax.random.split(rng)
            carry, pi = train_state.policy(
                train_state.params, carry, env_state.observation, env_state.is_done()
            )
            action = jnp.asarray(pi.sample(seed=_rng))
            log_prob = jnp.asarray(pi.log_prob(action))

            # STEP ENV
            new_env_state = jax.vmap(self.env.step, in_axes=(0, 0))(env_state, action)
            transition = Buffer(
                carry=carry,  # rnn state
                done=new_env_state.is_done(),  # done(o_{t+1})
                action=action,  # a_t
                reward=new_env_state.reward,  # R(o_t, a_t)
                log_prob=log_prob,  # log π(a_t|o_t)
                obs=env_state.observation,  # o_t
                info=new_env_state.info,  # info(o_{t+1})
                t=env_state.t,  # t
                state=env_state.state,  # s_t
            )
            return (new_env_state, rng, carry), transition

        # collect experience and update env_state
        (env_state, rng, carry), experience = jax.lax.scan(
            _env_step,
            (train_state.env_state, train_state.rng, train_state.carry),
            None,
            self.hparams.num_steps,
        )
        train_state = train_state.replace(
            carry=carry,
            env_state=env_state,
            rng=rng,
            frames=train_state.frames + self.hparams.num_steps * self.hparams.num_envs,
        )
        return train_state, experience

    def evaluate_experience(
        self, train_state: TrainingState, experience: Buffer, last_val: jax.Array
    ) -> Tuple[jax.Array, jax.Array, jax.Array, HiddenState]:
        carry, values = jnp.asarray(
            jax.vmap(train_state.value_fn, in_axes=(None, 0, 0))(
                train_state.params, experience.carry, experience.obs, experience.done
            )
        )  # (1:T, N)
        adv = jax.vmap(
            rlax.truncated_generalized_advantage_estimation,
            in_axes=(1, 1, None, 1, None),
            out_axes=1,
        )(
            experience.reward,  # (1:T, N)
            (1 - experience.done) * self.env.gamma**experience.t,  # (1:T, N)
            self.hparams.gae_lambda,  # ()
            jnp.concatenate([values, last_val[None]], axis=0),  # (0:T, N)
            True,
        )
        adv = jnp.asarray(adv)  # (0:T, N)
        targets = adv + values
        return values, adv, targets, carry

    def ppo_loss(
        self,
        params: Params,
        transition_batch: Buffer,
        gae: Array,
        targets: Array,
        values_old: Array,
        carry: HiddenState,
    ):
        # this is already vmapped over the minibatches
        pi, value = jax.vmap(self.network.apply, in_axes=(None, 0))(
            params, carry, transition_batch.obs
        )
        log_prob = pi.log_prob(transition_batch.action)

        # CALCULATE VALUE LOSS
        if self.hparams.clip_value_loss:
            value_loss = jnp.square(value - targets)
            value_clipped = values_old + jnp.clip(
                value - values_old,
                -self.hparams.clip_eps,
                self.hparams.clip_eps,
            )
            value_loss_clipped = 0.5 * jnp.square(value_clipped - targets)
            value_loss = 0.5 * jnp.maximum(value_loss, value_loss_clipped).mean()
        else:
            value_loss = 0.5 * jnp.square(value - targets).mean()

        # CALCULATE ACTOR LOSS
        ratio = jnp.exp(log_prob - transition_batch.log_prob)
        if self.hparams.normalise_advantage:
            gae = (gae - gae.mean()) / (gae.std() + 1e-8)
        loss_actor1 = ratio * gae
        loss_actor2 = (
            jnp.clip(
                ratio,
                1.0 - self.hparams.clip_eps,
                1.0 + self.hparams.clip_eps,
            )
            * gae
        )
        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
        loss_actor = loss_actor.mean()
        entropy = pi.entropy().mean()

        total_loss = (
            loss_actor
            + self.hparams.vf_coef * value_loss
            - self.hparams.ent_coef * entropy
        )

        # log
        logratio = log_prob - transition_batch.log_prob
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfrac = jnp.mean(jnp.abs(ratio - 1.0) > self.hparams.clip_eps)
        logs = {
            "loss/total_loss": total_loss,
            "loss/value_loss": value_loss,
            "loss/actor_loss": loss_actor,
            "loss/entropy": entropy,
            "loss/approx_kl": approx_kl,
            "loss/clipfrac": clipfrac,
        }
        return total_loss, logs

    def sgd_step(
        self,
        train_state: TrainingState,
        minibatch: Tuple[Buffer, jax.Array, jax.Array, jax.Array, HiddenState],
    ) -> Tuple[TrainingState, Dict]:
        traj_batch, advantages, targets, values_old, carry = minibatch
        grad_fn = jax.value_and_grad(self.ppo_loss, has_aux=True)
        (_, logs), grads = grad_fn(
            train_state.params, traj_batch, advantages, targets, values_old, carry
        )
        train_state = train_state.apply_gradients(grads=grads)
        return train_state, logs

    def update(self, train_state: TrainingState, _) -> Tuple[TrainingState, Dict]:
        # unpack state
        minibatch_size = (
            self.hparams.num_envs
            * self.hparams.num_steps
            // self.hparams.num_minibatches
        )
        # collect experience
        train_state, experience = self.collect_experience(train_state)

        for _ in range(self.hparams.num_epochs):
            # re-evaluate experience at every epoch as per https://arxiv.org/abs/2006.05990
            _, last_val = train_state.value_fn(
                train_state.params,
                train_state.carry,
                train_state.env_state.observation,
                experience.done,
            )  # boostrap
            values, advantages, targets, carry = self.evaluate_experience(
                train_state,
                experience,
                last_val,
            )

            # Batching and Shuffling
            rng, rng_1 = jax.random.split(train_state.rng)
            train_state = train_state.replace(rng=rng)
            n_samples = minibatch_size * self.hparams.num_minibatches
            assert (
                n_samples == self.hparams.num_steps * self.hparams.num_envs
            ), "batch size must be equal to number of steps * number of envs"
            permutation = jax.random.permutation(rng_1, n_samples)
            samples = (experience, advantages, targets, values, carry)  # (T, N, ...)
            samples = jax.tree_util.tree_map(
                lambda x: x.reshape((n_samples,) + x.shape[2:]), samples
            )  # (T * N, ...)
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0), samples
            )  # (T * N, ...)

            # One epoch update over all mini-batches
            minibatches = jax.tree_util.tree_map(
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

        # update logs with returns
        logs["done_mask"] = experience.done
        logs["returns"] = experience.info["return"]
        logs["lengths"] = experience.t
        logs["iter/frames"] = train_state.frames
        logs["iter/update_step"] = train_state.epoch
        logs["iter/train_step"] = train_state.step

        if self.hparams.log_render:
            b = jax.random.randint(rng, (), 0, self.hparams.num_envs)
            logs["render/human"] = jax.vmap(rgb)(
                jax.tree.map(lambda x: x[:, b], experience.state)
            ).transpose(
                (0, 3, 1, 2)
            )  # (T, 3, H, W)

        # Debugging mode
        if self.hparams.debug:
            jax.debug.callback(self.log, logs, experience)

        return train_state, logs

    def train(self, rng: jax.Array) -> Tuple[TrainingState, Dict]:
        # INIT NETWORK
        rng, _rng = jax.random.split(rng)
        obs = self.env.observation_space.sample(_rng)
        carry = self.network.initialize_carry(_rng, obs.shape)
        params = self.network.init(_rng, carry, obs, jnp.zeros(obs.shape[0]))

        def linear_schedule(count):
            frac = (
                1.0
                - (count // (self.hparams.num_minibatches * self.hparams.num_epochs))
                / num_updates
            )
            return self.hparams.lr * frac

        lr = linear_schedule if self.hparams.anneal_lr else self.hparams.lr
        tx = optax.chain(
            optax.clip_by_global_norm(self.hparams.max_grad_norm),
            optax.adam(lr, eps=1e-5),
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, self.hparams.num_envs)
        env_state = jax.vmap(self.env.reset)(reset_rng)

        # TRAIN LOOP
        num_updates = self.hparams.budget // (
            self.hparams.num_steps * self.hparams.num_envs
        )
        train_state = TrainingState.create(
            carry=carry,
            apply_fn=jax.vmap(self.network.apply, in_axes=(None, 0)),
            params=params,
            tx=tx,
            env_state=env_state,
            rng=rng,
            frames=jnp.asarray(0, dtype=jnp.int32),
            epoch=jnp.asarray(0, dtype=jnp.int32),
            policy=jax.vmap(
                partial(self.network.apply, method="policy"), in_axes=(None, 0)
            ),
            value_fn=jax.vmap(
                partial(self.network.apply, method="value"), in_axes=(None, 0)
            ),
        )
        train_state, logs = jax.lax.scan(self.update, train_state, length=num_updates)
        return train_state, logs
