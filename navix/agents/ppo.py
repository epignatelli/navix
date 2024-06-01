# This implementation of PPO is inspired from:
# https://github.com/luchris429/purejaxrl/blob/main/purejaxrl/ppo.py
from dataclasses import dataclass
import time

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Callable, Dict, NamedTuple, Tuple
from flax.training.train_state import TrainState
import distrax
import tyro
import wandb
from flax import struct

import navix as nx
from navix import observations
from navix.environments import Environment
from navix.environments.environment import Timestep


@dataclass
class Args:
    project_name = "navix"
    env_id: str = "Navix-DoorKey-Random-6x6-v0"
    seed: int = 0
    budget: int = 10_000_000
    debug: bool = True
    # ppo hyperparameters
    num_envs: int = 8
    num_steps: int = 256
    num_minibatches: int = 8
    update_epochs: int = 1
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    lr: float = 2.5e-4
    activation: str = "tanh"
    anneal_lr: bool = True


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


class UpdateState(NamedTuple):
    train_state: TrainingState
    traj_batch: Buffer
    advantages: jax.Array
    targets: jax.Array
    rng: jax.Array


class PPO(struct.PyTreeNode):
    hparams: Args = struct.field(pytree_node=False)
    network: ActorCritic = struct.field(pytree_node=False)
    env_step_fn: Callable[[Timestep, jax.Array], Timestep] = struct.field(
        pytree_node=False
    )

    def collect_experience(self, params, env_state, rng):
        def _env_step(
            collection_state: Tuple[Timestep, jax.Array], _
        ) -> Tuple[Tuple[Timestep, jax.Array], Buffer]:
            env_state, rng = collection_state
            # SELECT ACTION
            rng, _rng = jax.random.split(rng)
            pi, value = self.network.apply(params, env_state.observation)
            value = jnp.asarray(value)
            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            new_env_state = jax.vmap(self.env_step_fn, in_axes=(0, 0))(
                env_state, action
            )
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
            _env_step, (env_state, rng), None, self.hparams.num_steps
        )
        return (env_state, rng), experience

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
        log = {
            "loss/total_loss": total_loss,
            "loss/value_loss": value_loss,
            "loss/actor_loss": loss_actor,
            "loss/entropy": entropy,
            "loss/approx_kl": approx_kl,
            "loss/clipfrac": clipfrac,
        }
        return total_loss, log

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

    def update_step(self, update_state: UpdateState) -> Tuple[UpdateState, Dict]:
        # unpack state
        train_state = update_state.train_state
        traj_batch = update_state.traj_batch
        advantages = update_state.advantages
        targets = update_state.targets
        rng = update_state.rng
        minibatch_size = (
            self.hparams.num_envs
            * self.hparams.num_steps
            // self.hparams.num_minibatches
        )

        for _ in range(self.hparams.update_epochs):
            # Batching and Shuffling
            rng, _rng = jax.random.split(rng)
            batch_size = minibatch_size * self.hparams.num_minibatches
            assert (
                batch_size == self.hparams.num_steps * self.hparams.num_envs
            ), "batch size must be equal to number of steps * number of envs"
            permutation = jax.random.permutation(_rng, batch_size)
            batch = (traj_batch, advantages, targets)
            batch = jax.tree_util.tree_map(
                lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
            )
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0), batch
            )

            # One epoch update over all mini-batches
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.reshape(
                    x, [self.hparams.num_minibatches, -1] + list(x.shape[1:])
                ),
                shuffled_batch,
            )
            train_state, logs = jax.lax.scan(self.sgd_step, train_state, minibatches)

        update_state = UpdateState(
            train_state=train_state,
            traj_batch=traj_batch,
            advantages=advantages,
            targets=targets,
            rng=rng,
        )
        return update_state, logs

    def report_and_log(self, logs, experience):
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

    def report_final_log(self, logs):
        print(jax.tree.map(lambda x: x.shape, logs))
        len_logs = len(logs["update_step"])
        for step in range(len_logs):
            step_logs = {k: v[step] for k, v in logs.items()}
            self.report_and_log(step_logs, None)

    def update(self, train_state: TrainingState, _) -> Tuple[TrainingState, Dict]:
        # collect experience
        (env_state, rng), experience = self.collect_experience(
            params=train_state.params,
            env_state=train_state.env_state,
            rng=train_state.rng,
        )

        # evaluate experience
        last_val = jnp.asarray(
            self.network.apply(train_state.params, env_state.observation)[1]
        )
        advantages, targets = self.evaluate_experience(experience, last_val)

        # update agent
        update_state = UpdateState(
            train_state=train_state,
            traj_batch=experience,
            advantages=advantages,
            targets=targets,
            rng=train_state.rng,
        )
        update_state, logs = self.update_step(update_state)

        logs = jax.tree.map(lambda x: jnp.mean(x), logs)
        train_state = update_state.train_state
        rng = update_state.rng

        # update logs with returns
        logs["done_mask"] = experience.done
        logs["returns"] = experience.info["return"]
        logs["lengths"] = experience.t
        logs["frames"] = train_state.frames
        logs["update_step"] = train_state.epoch

        # Debugging mode
        if self.hparams.debug:
            jax.debug.callback(self.report_and_log, logs, experience)

        train_state = train_state.replace(
            train_state=train_state,
            env_state=env_state,
            rng=rng,
            frames=train_state.frames + self.hparams.num_steps * self.hparams.num_envs,
            epoch=train_state.epoch + self.hparams.update_epochs,
        )
        return train_state, logs

    def train(self, rng: jax.Array, *, args: Args) -> Tuple[TrainingState, Dict]:
        # INIT NETWORK
        network = ActorCritic(len(env.action_set), activation=args.activation)
        rng, _rng = jax.random.split(rng)
        init_x = env.observation_space.sample(_rng)
        network_params = network.init(_rng, init_x)
        linear_schedule = optax.linear_schedule(1.0, 0.0, args.budget)
        lr = linear_schedule if args.anneal_lr else args.lr
        tx = optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm), optax.adam(lr, eps=1e-5)
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, args.num_envs)
        env_state = jax.vmap(env.reset)(reset_rng)

        # TRAIN LOOP
        num_updates = args.budget // (args.num_steps * args.num_envs)
        train_state = TrainingState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
            env_state=env_state,
            rng=_rng,
            frames=jnp.asarray(0),
            epoch=jnp.asarray(0),
        )
        runner_state, logs = jax.lax.scan(agent.update, train_state, None, num_updates)
        return runner_state, logs


class Experiment:
    def __init__(self, agent: PPO, env: Environment, seed: int):
        self.agent = agent
        self.env = env
        self.seed = seed

    def run(self):
        rng = jax.random.PRNGKey(self.seed)

        print("Compiling training function...")
        start_time = time.time()
        train_fn = (
            jax.jit(self.agent.train, static_argnames=("args", "agent", "env"))
            .lower(rng)
            .compile()
        )
        compilation_time = time.time() - start_time
        print(f"Compilation time cost: {compilation_time}")

        print("Training agent...")
        start_time = time.time()
        train_state, logs = train_fn(rng)
        training_time = time.time() - start_time
        print(f"Training time cost: {training_time}")

        if not args.debug:
            print("Logging final results to wandb...")
            start_time = time.time()
            self.agent.report_final_log(logs)
            wandb.log({})
            logging_time = time.time() - start_time
            print(f"Logging time cost: {logging_time}")

        print("Training complete")
        print(f"Compilation time cost: {compilation_time}")
        print(f"Training time cost: {training_time}")
        total_time = compilation_time + training_time
        if not args.debug:
            print(f"Logging time cost: {logging_time}")
            total_time += logging_time
        print(f"Total time cost: {total_time}")
        return train_state, logs


if __name__ == "__main__":
    args = tyro.cli(Args)

    def FlattenObsWrapper(env: Environment):
        flatten_obs_fn = lambda x: jnp.ravel(env.observation_fn(x))
        flatten_obs_shape = (int(np.prod(env.observation_space.shape)),)
        return env.replace(
            observation_fn=flatten_obs_fn,
            observation_space=env.observation_space.replace(shape=flatten_obs_shape),
        )

    env = nx.make(
        args.env_id,
        max_steps=100,
        observation_fn=observations.symbolic,
    )
    env = FlattenObsWrapper(env)

    agent = PPO(
        hparams=args,
        network=ActorCritic(len(env.action_set), activation=args.activation),
        env_step_fn=env.step,
    )

    experiment = Experiment(agent=agent, env=env, seed=args.seed)
    train_state, logs = experiment.run()
