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

"""The carry-based encoder contract (`navix.agents.models`): every PPO
encoder exposes `initial_carry` + `__call__(carry, obs, is_first) ->
(carry, features)`. The stateless encoders (`MLPEncoder`, `ConvEncoder`)
must be inert w.r.t. the carry; `TransformerEncoder` (issue #169) carries
a rolling raw-frame window and produces a history-conditioned feature."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from navix.agents.models import (
    MLPEncoder,
    ConvEncoder,
    TransformerBlock,
    TransformerEncoder,
    ActorCritic,
)


HIDDEN = 8
CONTEXT = 4
FRAME = (5,)


def _transformer(**overrides) -> TransformerEncoder:
    kwargs = dict(
        frame_encoder=MLPEncoder(hidden_size=HIDDEN),
        hidden_size=HIDDEN,
        num_heads=2,
        num_layers=2,
        context=CONTEXT,
    )
    kwargs.update(overrides)
    return TransformerEncoder(**kwargs)


# --------------------------------------------------------------------------
# stateless encoders: carry is `()` and threading it is a no-op
# --------------------------------------------------------------------------


def test_stateless_encoders_carry_is_empty_and_passes_through():
    for enc in (MLPEncoder(hidden_size=HIDDEN), ConvEncoder(hidden_size=HIDDEN)):
        obs = jnp.zeros((6, 6, 3)) if isinstance(enc, ConvEncoder) else jnp.zeros((10,))
        assert enc.initial_carry(obs.shape) == ()
        params = enc.init(jax.random.PRNGKey(0), (), obs, jnp.asarray(False))
        carry, feats = enc.apply(params, (), obs, jnp.asarray(False))
        assert carry == ()
        assert feats.shape == (HIDDEN,)


def test_stateless_encoder_output_matches_plain_sequential():
    # the pre-carry MLPEncoder was `nn.Sequential([Dense, tanh, Dense,
    # tanh])(x)`; the carry version must return exactly that as its
    # feature, so a fixed-seed agent using it is unchanged.
    enc = MLPEncoder(hidden_size=HIDDEN)
    x = jax.random.normal(jax.random.PRNGKey(1), (10,))
    params = enc.init(jax.random.PRNGKey(0), (), x, jnp.asarray(False))
    _, feats = enc.apply(params, (), x, jnp.asarray(False))

    d0, d1 = params["params"]["Dense_0"], params["params"]["Dense_1"]
    h = jnp.tanh(x @ d0["kernel"] + d0["bias"])
    h = jnp.tanh(h @ d1["kernel"] + d1["bias"])
    assert np.allclose(feats, h, atol=1e-6)


# --------------------------------------------------------------------------
# TransformerBlock
# --------------------------------------------------------------------------


def test_transformer_block_preserves_shape():
    block = TransformerBlock(hidden_size=HIDDEN, num_heads=2)
    x = jax.random.normal(jax.random.PRNGKey(0), (CONTEXT, HIDDEN))
    params = block.init(jax.random.PRNGKey(1), x)
    assert block.apply(params, x).shape == x.shape


# --------------------------------------------------------------------------
# TransformerEncoder: shapes, carry contract, history sensitivity
# --------------------------------------------------------------------------


def test_transformer_encoder_shapes_and_carry_roundtrip():
    enc = _transformer()
    carry0 = enc.initial_carry(FRAME)
    assert carry0.shape == (CONTEXT, *FRAME)
    obs = jax.random.normal(jax.random.PRNGKey(0), FRAME)
    params = enc.init(jax.random.PRNGKey(1), carry0, obs, jnp.asarray(False))
    carry1, feats = enc.apply(params, carry0, obs, jnp.asarray(False))
    assert feats.shape == (HIDDEN,)
    assert carry1.shape == (CONTEXT, *FRAME)


def test_transformer_encoder_is_first_refills_window():
    # with is_first=True the incoming carry is discarded and the whole
    # window is set to `obs`, so the feature (and next carry) must match
    # what a zero carry would have produced.
    enc = _transformer()
    carry0 = enc.initial_carry(FRAME)
    obs = jax.random.normal(jax.random.PRNGKey(0), FRAME)
    params = enc.init(jax.random.PRNGKey(1), carry0, obs, jnp.asarray(False))

    filled = jax.random.normal(jax.random.PRNGKey(2), (CONTEXT, *FRAME))
    win_a, feat_a = enc.apply(params, filled, obs, jnp.asarray(True))
    win_b, feat_b = enc.apply(params, carry0, obs, jnp.asarray(True))

    assert np.allclose(win_a, jnp.broadcast_to(obs, win_a.shape))
    assert np.allclose(feat_a, feat_b)


def test_transformer_encoder_rolls_window_when_not_first():
    enc = _transformer()
    carry0 = enc.initial_carry(FRAME)
    obs = jax.random.normal(jax.random.PRNGKey(0), FRAME)
    params = enc.init(jax.random.PRNGKey(1), carry0, obs, jnp.asarray(False))
    filled = jax.random.normal(jax.random.PRNGKey(2), (CONTEXT, *FRAME))

    next_carry, _ = enc.apply(params, filled, obs, jnp.asarray(False))
    assert np.allclose(
        next_carry, jnp.concatenate([filled[1:], obs[None]], axis=0)
    )


def test_transformer_encoder_conditions_on_history_not_just_current_frame():
    # perturbing a frame that is actually inside the attention window
    # (the current obs, or any of the kept `context - 1` past frames)
    # must change the output - otherwise this has degenerated to a
    # single-frame encoder, defeating issue #169.
    enc = _transformer()
    carry0 = enc.initial_carry(FRAME)
    obs = jax.random.normal(jax.random.PRNGKey(0), FRAME)
    params = enc.init(jax.random.PRNGKey(1), carry0, obs, jnp.asarray(False))

    window = jax.random.normal(jax.random.PRNGKey(2), (CONTEXT, *FRAME))
    _, base = enc.apply(params, window, obs, jnp.asarray(False))
    # index 0 of the incoming carry is evicted by the roll; 1..context-1
    # are the kept past frames.
    for i in range(1, CONTEXT):
        perturbed = window.at[i].set(
            jax.random.normal(jax.random.PRNGKey(10 + i), FRAME)
        )
        _, feat = enc.apply(params, perturbed, obs, jnp.asarray(False))
        assert not np.allclose(base, feat), f"kept past frame {i} had no effect"


def test_transformer_encoder_shares_frame_encoder_weights():
    # one MLPEncoder's worth of leaves (2 Dense -> 4), regardless of
    # `context` - not `context` independent copies.
    enc = _transformer(context=6)
    carry0 = enc.initial_carry(FRAME)
    obs = jax.random.normal(jax.random.PRNGKey(0), FRAME)
    params = enc.init(jax.random.PRNGKey(1), carry0, obs, jnp.asarray(False))
    leaves = jax.tree_util.tree_leaves(params["params"]["frame_encoder"])
    assert len(leaves) == 4


def test_transformer_encoder_jit_vmap_over_batch():
    enc = _transformer()
    carry0 = enc.initial_carry(FRAME)
    obs = jax.random.normal(jax.random.PRNGKey(0), FRAME)
    params = enc.init(jax.random.PRNGKey(1), carry0, obs, jnp.asarray(False))

    batch = 3
    carry_b = jnp.broadcast_to(carry0, (batch, *carry0.shape))
    obs_b = jax.random.normal(jax.random.PRNGKey(2), (batch, *FRAME))
    first_b = jnp.asarray([True, False, True])
    fn = jax.jit(jax.vmap(enc.apply, in_axes=(None, 0, 0, 0)))
    carry_out, feats = fn(params, carry_b, obs_b, first_b)
    assert carry_out.shape == (batch, CONTEXT, *FRAME)
    assert feats.shape == (batch, HIDDEN)


# --------------------------------------------------------------------------
# ActorCritic threads a per-encoder carry tuple
# --------------------------------------------------------------------------


def test_actor_critic_carry_is_empty_for_stateless_encoders():
    net = ActorCritic(action_dim=4)
    assert net.initial_carry((10,)) == ()
    x = jnp.zeros((10,))
    params = net.init(jax.random.PRNGKey(0), (), x, jnp.asarray(False))
    carry, (pi, value) = net.apply(params, (), x, jnp.asarray(False))
    assert carry == ()
    assert pi.logits.shape == (4,)
    assert value.shape == ()


def test_actor_critic_with_transformer_encoders_threads_one_shared_window_carry():
    # actor and critic share a single frame-window carry (they derive it
    # identically from the obs stream) - so it's advanced once per step
    # even when only `policy` runs.
    net = ActorCritic(
        action_dim=4,
        actor_encoder=_transformer(),
        critic_encoder=_transformer(),
    )
    carry0 = net.initial_carry(FRAME)
    assert carry0.shape == (CONTEXT, *FRAME)
    x = jax.random.normal(jax.random.PRNGKey(0), FRAME)
    params = net.init(jax.random.PRNGKey(1), carry0, x, jnp.asarray(False))
    carry1, (pi, value) = net.apply(params, carry0, x, jnp.asarray(False))
    assert carry1.shape == (CONTEXT, *FRAME)
    assert pi.logits.shape == (4,)
    assert value.shape == ()
    # `policy` and `value` advance the shared carry the same way `__call__` does
    pcarry, _ = net.apply(params, carry0, x, jnp.asarray(False), method="policy")
    vcarry, _ = net.apply(params, carry0, x, jnp.asarray(False), method="value")
    assert np.allclose(pcarry, carry1) and np.allclose(vcarry, carry1)


if __name__ == "__main__":
    test_stateless_encoders_carry_is_empty_and_passes_through()
    test_stateless_encoder_output_matches_plain_sequential()
    test_transformer_block_preserves_shape()
    test_transformer_encoder_shapes_and_carry_roundtrip()
    test_transformer_encoder_is_first_refills_window()
    test_transformer_encoder_rolls_window_when_not_first()
    test_transformer_encoder_conditions_on_history_not_just_current_frame()
    test_transformer_encoder_shares_frame_encoder_weights()
    test_transformer_encoder_jit_vmap_over_batch()
    test_actor_critic_carry_is_empty_for_stateless_encoders()
    test_actor_critic_with_transformer_encoders_threads_one_shared_window_carry()
