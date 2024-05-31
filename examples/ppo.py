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
from typing import Dict, NamedTuple, Tuple
from flax.training.train_state import TrainState
import distrax
import tyro
import wandb

import navix as nx
from navix import observations
from navix.environments import Environment
from navix.environments.environment import Timestep


@dataclass
class Args:
    project_name = "navix"
    env_id: str = "Navix-Empty-Random-5x5-v0"
    seed: int = 0
    budget: int = 10_000_000
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
    debug: bool = True


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


def FlattenObsWrapper(env: Environment):
    flatten_obs_fn = lambda x: jnp.ravel(env.observation_fn(x))
    flatten_obs_shape = (int(np.prod(env.observation_space.shape)),)
    return env.replace(
        observation_fn=flatten_obs_fn,
        observation_space=env.observation_space.replace(shape=flatten_obs_shape),
    )


class Buffer(NamedTuple):
    done: jax.Array
    action: jax.Array
    value: jax.Array
    reward: jax.Array
    log_prob: jax.Array
    obs: jax.Array
    info: Dict[str, jax.Array]
    t: jax.Array


class RunnerState(NamedTuple):
    train_state: TrainState
    env_state: Timestep
    rng: jax.Array
    frames: jax.Array
    updates: jax.Array


class UpdateState(NamedTuple):
    train_state: TrainState
    traj_batch: Buffer
    advantages: jax.Array
    targets: jax.Array
    rng: jax.Array


def make_train(args):
    wandb.init(project=args.project_name)

    env = nx.make(
        args.env_id,
        max_steps=100,
        observation_fn=observations.symbolic,
    )
    # flatten obs
    env = FlattenObsWrapper(env)

    num_updates = args.budget // (args.num_steps * args.num_envs)
    minibatch_size = args.num_envs * args.num_steps // args.num_minibatches

    def linear_schedule(count):
        frac = (
            1.0 - (count // (args.num_minibatches * args.update_epochs)) / num_updates
        )
        return args.lr * frac

    def train(rng):
        # INIT NETWORK
        network = ActorCritic(len(env.action_set), activation=args.activation)
        rng, _rng = jax.random.split(rng)
        init_x = env.observation_space.sample(_rng)
        network_params = network.init(_rng, init_x)
        lr = linear_schedule if args.anneal_lr else args.lr
        tx = optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm), optax.adam(lr, eps=1e-5)
        )

        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, args.num_envs)
        env_state = jax.vmap(env.reset)(reset_rng)

        # TRAIN LOOP
        def _update_step(runner_state: RunnerState, _) -> Tuple[RunnerState, Dict]:
            # unpack state
            train_state = runner_state.train_state
            env_state = runner_state.env_state
            rng = runner_state.rng
            frames = runner_state.frames
            updates = runner_state.updates

            # -----------------------------------------------------------------
            # --- COLLECT EXPERIENCE ------------------------------------------
            # -----------------------------------------------------------------
            def _env_step(
                collection_state: Tuple[Timestep, jax.Array], _
            ) -> Tuple[Tuple[Timestep, jax.Array], Buffer]:
                env_state, rng = collection_state
                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, env_state.observation)
                value = jnp.asarray(value)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                new_env_state = jax.vmap(env.step, in_axes=(0, 0))(env_state, action)
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
                _env_step, (env_state, rng), None, args.num_steps
            )
            # update number of elapsed frames
            frames += args.num_steps * args.num_envs
            # -----------------------------------------------------------------

            # -----------------------------------------------------------------
            # --- EVALUATE EXPERIENCE -----------------------------------------
            # -----------------------------------------------------------------
            def evaluate_experience(
                experience: Buffer, last_val: jax.Array
            ) -> Tuple[jax.Array, jax.Array]:
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + args.gamma * next_value * (1 - done) - value
                    gae = delta + args.gamma * args.gae_lambda * (1 - done) * gae
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    experience,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + experience.value

            last_val = jnp.asarray(
                network.apply(runner_state.train_state.params, env_state.observation)[1]
            )
            advantages, targets = evaluate_experience(experience, last_val)
            # -----------------------------------------------------------------

            # -----------------------------------------------------------------
            # --- UPDATE AGENT  -----------------------------------------------
            # -----------------------------------------------------------------
            def _update_epoch(update_state: UpdateState, _) -> Tuple[UpdateState, Dict]:
                def _update_minbatch(
                    train_state: TrainState,
                    minibatch: Tuple[Buffer, jax.Array, jax.Array],
                ) -> Tuple[TrainState, Dict]:
                    traj_batch, advantages, targets = minibatch

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-args.clip_eps, args.clip_eps)
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - args.clip_eps,
                                1.0 + args.clip_eps,
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + args.vf_coef * value_loss
                            - args.ent_coef * entropy
                        )

                        # log
                        logratio = log_prob - traj_batch.log_prob
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfrac = jnp.mean(jnp.abs(ratio - 1.0) > args.clip_eps)
                        log = {
                            "ppo/total_loss": total_loss,
                            "ppo/value_loss": value_loss,
                            "ppo/actor_loss": loss_actor,
                            "ppo/entropy": entropy,
                            "ppo/approx_kl": approx_kl,
                            "ppo/clipfrac": clipfrac,
                        }
                        return total_loss, log

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    (_, log), grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, log

                train_state = update_state.train_state
                traj_batch = update_state.traj_batch
                advantages = update_state.advantages
                targets = update_state.targets
                rng = update_state.rng
                rng, _rng = jax.random.split(rng)

                # Batching and Shuffling
                batch_size = minibatch_size * args.num_minibatches
                assert (
                    batch_size == args.num_steps * args.num_envs
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )

                # Mini-batch Updates
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [args.num_minibatches, -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, logs = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = UpdateState(
                    train_state=train_state,
                    traj_batch=traj_batch,
                    advantages=advantages,
                    targets=targets,
                    rng=rng,
                )
                return update_state, logs

            # init state
            update_state = UpdateState(
                train_state=runner_state.train_state,
                traj_batch=experience,
                advantages=advantages,
                targets=targets,
                rng=rng,
            )
            # update state
            update_state, logs = jax.lax.scan(
                _update_epoch, update_state, None, args.update_epochs
            )
            train_state = update_state.train_state
            rng = update_state.rng

            # update logs with returns
            logs["done_mask"] = experience.done
            logs["returns"] = experience.info["return"]
            logs["lengths"] = experience.t
            logs["frames"] = runner_state.frames
            logs["update_step"] = runner_state.updates

            # Debugging mode
            if args.debug:
                jax.debug.callback(report_and_log, logs)

            runner_state = RunnerState(
                train_state=train_state,
                env_state=env_state,
                rng=rng,
                frames=frames,
                updates=updates + args.update_epochs,
            )
            return runner_state, logs

        rng, _rng = jax.random.split(rng)
        runner_state = RunnerState(
            train_state=train_state,
            env_state=env_state,
            rng=_rng,
            frames=jnp.asarray(0),
            updates=jnp.asarray(0),
        )
        runner_state, logs = jax.lax.scan(_update_step, runner_state, None, num_updates)
        return runner_state, logs
        # -----------------------------------------------------------------

    return train


def report_and_log(logs, commit=True):
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
    wandb.log(logs, step=logs["update_step"], commit=commit)


def report_final_log(logs):
    len_logs = len(logs["update_step"])
    for step in range(len_logs):
        step_logs = {k: v[step] for k, v in logs.items()}
        report_and_log(step_logs, commit=True)


if __name__ == "__main__":
    args = tyro.cli(Args)
    rng = jax.random.PRNGKey(args.seed)

    print("Compiling training function...")
    start_time = time.time()
    train_fn = jax.jit(make_train(args)).lower(rng).compile()
    compilation_time = time.time() - start_time
    print(f"Compilation time cost: {compilation_time}")

    print("Training agent...")
    start_time = time.time()
    runner_state, logs = train_fn(rng)
    training_time = time.time() - start_time
    print(f"Training time cost: {training_time}")

    if not args.debug:
        print("Logging final results to wandb...")
        start_time = time.time()
        report_final_log(logs)
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
