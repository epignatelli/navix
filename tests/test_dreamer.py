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

from unittest.mock import patch

import numpy as np
import jax
import jax.numpy as jnp

import navix as nx
from navix.agents.agent import Agent
from navix.agents.dreamer import (
    Dreamer,
    DreamerHparams,
    WorldModel,
    Actor,
    Critic,
)


def _make_dreamer(**hparam_overrides) -> Dreamer:
    # A deliberately tiny configuration - just enough for every code path
    # (collection, world-model/actor/critic updates, imagination rollouts,
    # sequence sampling) to actually execute, not to train anything useful.
    hp = DreamerHparams(
        budget=hparam_overrides.pop("budget", 256),
        num_envs=hparam_overrides.pop("num_envs", 4),
        num_steps=hparam_overrides.pop("num_steps", 8),
        num_model_updates=hparam_overrides.pop("num_model_updates", 2),
        num_actor_updates=hparam_overrides.pop("num_actor_updates", 2),
        num_critic_updates=hparam_overrides.pop("num_critic_updates", 2),
        batch_size=hparam_overrides.pop("batch_size", 3),
        seq_len=hparam_overrides.pop("seq_len", 4),
        imag_horizon=hparam_overrides.pop("imag_horizon", 3),
        embed_size=hparam_overrides.pop("embed_size", 8),
        deter_size=hparam_overrides.pop("deter_size", 8),
        stoch_size=hparam_overrides.pop("stoch_size", 4),
        hidden_size=hparam_overrides.pop("hidden_size", 8),
        **hparam_overrides,
    )
    env = nx.make("Navix-Empty-5x5-v0", max_steps=hp.num_steps)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = len(env.action_set)
    world = WorldModel(obs_dim=obs_dim, act_dim=act_dim, hparams=hp)
    actor = Actor(act_dim=act_dim, hidden=hp.hidden_size)
    critic = Critic(hidden=hp.hidden_size)
    return Dreamer(hparams=hp, env=env, world=world, actor=actor, critic=critic)


def test_dreamer_is_an_agent():
    # follows the Agent interface: HParams subclass, hparams field, and
    # inherits (rather than reimplements) the wandb logging machinery -
    # PPO does the same, this keeps both algorithms interchangeable from
    # Experiment's point of view.
    dreamer = _make_dreamer()
    assert isinstance(dreamer, Agent)
    assert isinstance(dreamer.hparams, DreamerHparams)
    assert hasattr(dreamer, "train")
    assert Dreamer.log_to_wandb is Agent.log_to_wandb
    assert Dreamer.log_to_wandb_on_train_end is Agent.log_to_wandb_on_train_end


def test_dreamer_trains_one_update_without_nans():
    dreamer = _make_dreamer(budget=8 * 4)  # -> exactly 1 update
    ts, logs = jax.jit(dreamer.train)(jax.random.PRNGKey(0))

    assert int(ts.updates) == 1, "expected exactly one Dreamer.update() call"

    for key in ("iter/frames", "iter/updates", "done_mask", "returns", "lengths"):
        assert key in logs, f"missing expected log key {key!r}"

    for key, value in logs.items():
        if key in ("done_mask",):
            continue
        arr = np.asarray(value)
        assert np.all(np.isfinite(arr)), f"logs[{key!r}] contains non-finite values"


def test_dreamer_networks_have_independent_optimizers_and_step_counters():
    # Regression test for the bug found while porting this agent from the
    # neurips branch: DreamerTrainState used to subclass flax's TrainState
    # directly and hand-roll tx.update() + optax.apply_updates() for all
    # three networks while sharing one `tx`/`step` field meant to be
    # swapped between updates. The swap never took effect before each
    # network's own update ran, so actor and critic gradients were
    # silently applied through the model's optimizer (and learning rate)
    # instead of their own - and since apply_gradients() was never called
    # at all, the step counter never incremented, breaking
    # log_frequency-gated logging.
    #
    # With three independent TrainStates, each network's .step must equal
    # exactly its own configured number of updates after one
    # Dreamer.update() call - this is only possible if each one is really
    # calling its own apply_gradients(), not sharing state with another.
    dreamer = _make_dreamer(
        num_model_updates=3, num_actor_updates=5, num_critic_updates=7
    )
    ts, _ = jax.jit(dreamer.train_first_update)(jax.random.PRNGKey(0))

    assert int(ts.model.step) == 3
    assert int(ts.actor.step) == 5
    assert int(ts.critic.step) == 7
    # if the three optimizers were accidentally the same object/state, the
    # three step counts above could not differ from one another the way
    # they do here (3 != 5 != 7) - this is only possible with genuinely
    # independent TrainStates.


def test_dreamer_logs_flow_through_agent_log_to_wandb():
    # smoke test that Dreamer's logs dict is shaped the way the base
    # Agent's wandb-logging methods expect (done_mask/returns/lengths ->
    # perf/* via masked_mean), the same contract PPO relies on.
    dreamer = _make_dreamer(budget=8 * 4)
    ts, logs = jax.jit(dreamer.train)(jax.random.PRNGKey(0))
    logs = jax.tree.map(lambda x: x[0] if hasattr(x, "shape") and x.shape else x, logs)

    with patch("navix.agents.agent.wandb.log") as mock_log:
        dreamer.log_to_wandb(dict(logs))

    assert mock_log.call_count == 1
    logged = mock_log.call_args.args[0]
    assert "perf/returns" in logged
    assert "perf/episode_length" in logged
    assert "perf/success_rate" in logged


if __name__ == "__main__":
    test_dreamer_is_an_agent()
    test_dreamer_trains_one_update_without_nans()
    test_dreamer_networks_have_independent_optimizers_and_step_counters()
    test_dreamer_logs_flow_through_agent_log_to_wandb()
