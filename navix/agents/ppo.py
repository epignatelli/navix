# This implementation of PPO is roughly inspired by:
# https://github.com/luchris429/purejaxrl/blob/main/purejaxrl/ppo.py
from dataclasses import dataclass, asdict
import time

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Dict, NamedTuple, Tuple
from flax.training.train_state import TrainState
import distrax
import wandb
from flax import struct

from navix.agents.agent import Agent, HParams
from navix.environments import Environment
from navix.environments.environment import Timestep


# THIS DOES NOT WORK!
# See https://github.com/google/flax/issues/3956
# class ActorCritic(nn.Module):
#     actor_encoder: nn.Module
#     critic_encoder: nn.Module
#     action_dim: int

#     @nn.compact
#     def __call__(self, x):
#         actor_repr = self.actor_encoder(x)
#         logits = nn.Dense(
#             self.action_dim,
#             kernel_init=orthogonal(0.01),
#             bias_init=constant(0.0),
#         )(actor_repr)
#         pi = distrax.Categorical(logits=logits)

#         critic_repr = self.critic_encoder(x)
#         value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
#             critic_repr
#         )
#         return pi, jnp.squeeze(value, axis=-1)


@dataclass
class PPOHparams(HParams):
    budget: int = 10_000_000
    """Number of environment frames to train for."""
    num_envs: int = 8
    """Number of parallel environments to run."""
    num_steps: int = 256
    """Number of steps to run in each environment per update."""
    num_minibatches: int = 8
    """Number of minibatches to split the data into for training."""
    num_epochs: int = 1
    """Number of epochs to train for."""
    gamma: float = 0.99
    """Discount factor."""
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
    debug: bool = True
    """Whether to run in debug mode."""


class ActorCritic(nn.Module):
    action_dim: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        activation = getattr(nn, self.activation)
        n = self.action_dim
        logits = nn.Sequential(
            [
                nn.Dense(
                    64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
                ),
                activation,
                nn.Dense(
                    64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
                ),
                activation,
                nn.Dense(n, kernel_init=orthogonal(0.01), bias_init=constant(0.0)),
            ]
        )(x)
        pi = distrax.Categorical(logits=logits)

        value = nn.Sequential(
            [
                nn.Dense(
                    64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
                ),
                activation,
                nn.Dense(
                    64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
                ),
                activation,
                nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0)),
            ]
        )(x)

        return pi, jnp.squeeze(value, axis=-1)


class Buffer(NamedTuple):
    done: jax.Array
    action: jax.Array
    value: jax.Array
    reward: jax.Array
    log_prob: jax.Array
    obs: jax.Array
    info: Dict[str, jax.Array]
    t: jax.Array


class TrainingState(TrainState):
    env_state: Timestep
    rng: jax.Array
    frames: jax.Array
    epoch: jax.Array


class PPO(Agent):
    hparams: PPOHparams = struct.field(pytree_node=False)
    network: ActorCritic = struct.field(pytree_node=False)
    env: Environment

    def collect_experience(
        self, train_state: TrainingState
    ) -> Tuple[TrainingState, Buffer]:
        def _env_step(
            collection_state: Tuple[Timestep, jax.Array], _
        ) -> Tuple[Tuple[Timestep, jax.Array], Buffer]:
            env_state, rng = collection_state
            # SELECT ACTION
            rng, _rng = jax.random.split(rng)
            pi, value = self.network.apply(train_state.params, env_state.observation)
            value = jnp.asarray(value)
            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            new_env_state = jax.vmap(self.env.step, in_axes=(0, 0))(env_state, action)
            transition = Buffer(
                new_env_state.is_done(),
                action,
                value,
                new_env_state.reward,
                log_prob,
                new_env_state.observation,
                new_env_state.info,
                env_state.t,
            )
            return (new_env_state, rng), transition

        # collect experience and update env_state
        (env_state, rng), experience = jax.lax.scan(
            _env_step,
            (train_state.env_state, train_state.rng),
            None,
            self.hparams.num_steps,
        )
        train_state = train_state.replace(
            env_state=env_state,
            rng=rng,
            frames=train_state.frames + self.hparams.num_steps * self.hparams.num_envs,
        )
        return train_state, experience

    def evaluate_experience(
        self, experience: Buffer, last_val: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        def _get_advantages(gae_and_next_value, transition):
            gae, next_value = gae_and_next_value
            done, value, reward = (
                transition.done,
                transition.value,
                transition.reward,
            )
            delta = reward + self.hparams.gamma * next_value * (1 - done) - value
            gae = (
                delta + self.hparams.gamma * self.hparams.gae_lambda * (1 - done) * gae
            )
            return (gae, value), gae

        _, advantages = jax.lax.scan(
            _get_advantages,
            (jnp.zeros_like(last_val), last_val),
            experience,
            reverse=True,
            unroll=16,
        )
        return advantages, advantages + experience.value

    def ppo_loss(self, params, transition_batch, gae, targets):
        # this is already vmapped over the minibatches
        # RERUN NETWORK
        pi, value = self.network.apply(params, transition_batch.obs)
        log_prob = pi.log_prob(transition_batch.action)

        # CALCULATE VALUE LOSS
        value_pred_clipped = transition_batch.value + (
            value - transition_batch.value
        ).clip(-self.hparams.clip_eps, self.hparams.clip_eps)
        value_losses = jnp.square(value - targets)
        value_losses_clipped = jnp.square(value_pred_clipped - targets)
        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

        # CALCULATE ACTOR LOSS
        ratio = jnp.exp(log_prob - transition_batch.log_prob)
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
        minibatch: Tuple[Buffer, jax.Array, jax.Array],
    ) -> Tuple[TrainingState, Dict]:
        traj_batch, advantages, targets = minibatch
        grad_fn = jax.value_and_grad(self.ppo_loss, has_aux=True)
        (_, log), grads = grad_fn(train_state.params, traj_batch, advantages, targets)
        train_state = train_state.apply_gradients(grads=grads)
        return train_state, log

    def update(self, train_state: TrainingState, _) -> Tuple[TrainingState, Dict]:
        # unpack state
        rng = train_state.rng
        minibatch_size = (
            self.hparams.num_envs
            * self.hparams.num_steps
            // self.hparams.num_minibatches
        )
        # collect experience
        train_state, experience = self.collect_experience(train_state)

        last_obs = train_state.env_state.observation
        for _ in range(self.hparams.num_epochs):
            # Re-evaluate experience at every epoch as per
            # https://arxiv.org/abs/2006.05990
            last_val = jnp.asarray(self.network.apply(train_state.params, last_obs)[1])
            advantages, targets = self.evaluate_experience(experience, last_val)

            # Batching and Shuffling
            rng, rng_1 = jax.random.split(rng)
            batch_size = minibatch_size * self.hparams.num_minibatches
            assert (
                batch_size == self.hparams.num_steps * self.hparams.num_envs
            ), "batch size must be equal to number of steps * number of envs"
            permutation = jax.random.permutation(rng_1, batch_size)
            batch = (experience, advantages, targets)
            batch = jax.tree_util.tree_map(
                lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
            )
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0), batch
            )

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
        logs["frames"] = train_state.frames
        logs["update_step"] = train_state.epoch
        logs["train_step"] = train_state.step

        # Debugging mode
        if self.hparams.debug:
            jax.debug.callback(self.log, logs, experience)

        return train_state, logs

    def train(self, rng: jax.Array) -> Tuple[TrainingState, Dict]:
        # INIT NETWORK
        rng, _rng = jax.random.split(rng)
        init_x = self.env.observation_space.sample(_rng)
        network_params = self.network.init(_rng, init_x)

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
            apply_fn=self.network.apply,
            params=network_params,
            tx=tx,
            env_state=env_state,
            rng=_rng,
            frames=jnp.asarray(0),
            epoch=jnp.asarray(0),
        )
        train_state, logs = jax.lax.scan(self.update, train_state, length=num_updates)
        return train_state, logs

    def log(self, logs, inspectable):
        if len(logs) == 0:
            return
        start_time = time.time()
        mask = logs.pop("done_mask")  # (T, N)
        returns = logs.pop("returns")  # (T, N)
        lengths = logs.pop("lengths")  # (T, N)

        # mask out incomplete episodes
        final_returns = returns[mask]  # (K,)
        episode_lengths = lengths[mask]  # (K,)

        logs["perf/returns"] = jnp.mean(final_returns)
        logs["perf/episode_length"] = jnp.mean(episode_lengths)
        logs["perf/success_rate"] = jnp.mean(final_returns == 1.0)

        print(
            (
                f"Update Step: {logs['update_step']}, "
                f"Frames: {logs['frames']}, "
                f"Returns: {logs['perf/returns']}, "
                f"Length: {logs['perf/episode_length']}, "
                f"Success Rate: {logs['perf/success_rate']}, "
                f"Logging time cost: {time.time() - start_time}"
            )
        )

        logs = {k: v.item() for k, v in logs.items()}
        wandb.log(logs, step=logs["update_step"])

    def log_on_train_end(self, logs):
        print(jax.tree.map(lambda x: x.shape, logs))
        len_logs = len(logs["update_step"])
        for step in range(len_logs):
            step_logs = {k: v[step] for k, v in logs.items()}
            self.log(step_logs, None)
