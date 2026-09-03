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

"""`PPO` under the carry-based encoder contract: a stateless encoder
leaves the training loop numerically unchanged (carry is `()`), and
swapping in `TransformerEncoder` (issue #169) - and nothing else - trains
a history-conditioned policy through the same loop."""

import numpy as np
import jax
import jax.numpy as jnp

import navix as nx
from navix import observations
from navix.agents.agent import Agent
from navix.agents.ppo import PPO, PPOHparams
from navix.agents.models import ActorCritic, MLPEncoder, TransformerEncoder


def _flatten(env):
    fn = lambda state: jnp.ravel(env.observation_fn(state))
    shape = (int(np.prod(env.observation_space.shape)),)
    return env.replace(
        observation_fn=fn, observation_space=env.observation_space.replace(shape=shape)
    )


def _hparams(**overrides) -> PPOHparams:
    # Tiny - just enough for every code path (collection, GAE, minibatch
    # SGD, bootstrap) to execute. budget = num_steps * num_envs -> 1 update.
    return PPOHparams(
        budget=overrides.pop("budget", 8 * 4),
        num_envs=overrides.pop("num_envs", 4),
        num_steps=overrides.pop("num_steps", 8),
        num_minibatches=overrides.pop("num_minibatches", 2),
        num_epochs=overrides.pop("num_epochs", 2),
        **overrides,
    )


def _make_ppo(network, **hp) -> PPO:
    hparams = _hparams(**hp)
    env = _flatten(
        nx.make(
            "Navix-Empty-5x5-v0",
            observation_fn=observations.symbolic_first_person,
            max_steps=hparams.num_steps,
        )
    )
    return PPO(hparams=hparams, network=network, env=env)


def _transformer_actor_critic(action_dim: int, context: int = 4) -> ActorCritic:
    enc = lambda: TransformerEncoder(
        frame_encoder=MLPEncoder(hidden_size=16),
        hidden_size=16,
        num_heads=2,
        num_layers=2,
        context=context,
    )
    return ActorCritic(action_dim=action_dim, actor_encoder=enc(), critic_encoder=enc())


def test_ppo_is_an_agent():
    ppo = _make_ppo(ActorCritic(action_dim=7))
    assert isinstance(ppo, Agent)
    assert isinstance(ppo.hparams, PPOHparams)
    assert PPO.log_to_wandb is Agent.log_to_wandb


def test_ppo_stateless_encoder_trains_one_update_without_nans():
    ppo = _make_ppo(ActorCritic(action_dim=7))
    ts, logs = jax.jit(ppo.train)(jax.random.PRNGKey(0))
    for key in (
        "agent/train/frames",
        "agent/train/updates",
        "agent/train/done_mask",
        "agent/train/returns",
        "agent/diagnostics/total_loss",
    ):
        assert key in logs, f"missing {key!r}"
    for key, value in logs.items():
        if key == "agent/train/done_mask":
            continue
        assert np.all(np.isfinite(np.asarray(value))), f"non-finite in logs[{key!r}]"


def test_ppo_stateless_encoder_carry_is_empty_end_to_end():
    # a stateless encoder must thread `()` everywhere: the live carry and
    # the rollout-initial carry in TrainingState are both empty pytrees.
    ppo = _make_ppo(ActorCritic(action_dim=7))
    ts = _init_state(ppo, jax.random.PRNGKey(0))
    assert jax.tree_util.tree_leaves(ts.carry) == []
    assert jax.tree_util.tree_leaves(ts.rollout_carry) == []
    ts, _ = jax.jit(ppo.collect_experience)(ts)
    assert jax.tree_util.tree_leaves(ts.rollout_carry) == []


def test_ppo_transformer_encoder_trains_one_update_without_nans():
    ppo = _make_ppo(_transformer_actor_critic(action_dim=7, context=4))
    ts, logs = jax.jit(ppo.train)(jax.random.PRNGKey(0))
    assert np.all(np.isfinite(np.asarray(logs["agent/diagnostics/total_loss"])))
    assert np.all(np.isfinite(np.asarray(logs["agent/diagnostics/value_loss"])))
    # the attention stack actually got parameters
    paths = [
        "/".join(str(k.key) for k in p)
        for p, _ in jax.tree_util.tree_leaves_with_path(ts.params)
    ]
    assert any("MultiHeadDotProductAttention" in p for p in paths)
    assert any("pos_embedding" in p for p in paths)


def test_ppo_transformer_encoder_tracks_rollout_initial_window():
    ppo = _make_ppo(_transformer_actor_critic(action_dim=7, context=4))
    ts = _init_state(ppo, jax.random.PRNGKey(0))
    ts, _ = jax.jit(ppo.collect_experience)(ts)
    # rollout_carry: (actor_window, critic_window), each the per-env frame
    # window (num_envs, context, *frame_shape) the rescan starts from.
    actor_window = ts.rollout_carry[0]
    frame_dim = int(np.prod(ppo.env.observation_space.shape))
    assert actor_window.shape == (ppo.hparams.num_envs, 4, frame_dim)


def _init_state(ppo: PPO, rng):
    # The part of PPO.train needed to get a real TrainingState without
    # running the whole training scan (mirrors tests/test_pqn.py).
    import optax
    from functools import partial
    from navix.agents.ppo import TrainingState

    rng, rng_init, rng_env = jax.random.split(rng, 3)
    init_x = ppo.env.observation_space.sample(rng_init)
    carry_single = ppo.network.initial_carry(init_x.shape)
    params = ppo.network.init(rng_init, carry_single, init_x, jnp.asarray(False))
    tx = optax.chain(
        optax.clip_by_global_norm(ppo.hparams.max_grad_norm),
        optax.inject_hyperparams(optax.adam)(learning_rate=ppo.hparams.lr, eps=1e-5),
    )
    env_state = jax.vmap(ppo.env.reset)(jax.random.split(rng_env, ppo.hparams.num_envs))
    carry = jax.tree.map(
        lambda c: jnp.broadcast_to(c, (ppo.hparams.num_envs, *c.shape)), carry_single
    )
    return TrainingState.create(
        apply_fn=jax.vmap(ppo.network.apply, in_axes=(None, 0, 0, 0)),
        params=params,
        tx=tx,
        env_state=env_state,
        rng=rng,
        carry=carry,
        rollout_carry=carry,
        frames=jnp.asarray(0, dtype=jnp.int32),
        epoch=jnp.asarray(0, dtype=jnp.int32),
        policy=jax.vmap(
            partial(ppo.network.apply, method="policy"), in_axes=(None, 0, 0, 0)
        ),
        value_fn=jax.vmap(
            partial(ppo.network.apply, method="value"), in_axes=(None, 0, 0, 0)
        ),
    )


if __name__ == "__main__":
    test_ppo_is_an_agent()
    test_ppo_stateless_encoder_trains_one_update_without_nans()
    test_ppo_stateless_encoder_carry_is_empty_end_to_end()
    test_ppo_transformer_encoder_trains_one_update_without_nans()
    test_ppo_transformer_encoder_tracks_rollout_initial_window()
