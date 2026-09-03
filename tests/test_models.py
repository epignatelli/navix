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

"""Issue #169: `TransformerEncoder`/`TransformerBlock` (a frame-history
encoder for POMDP tasks) and `stack_frame_history` (the trajectory-
windowing utility that feeds it - tested in `tests/test_agents.py`,
alongside `masked_mean`/`derive_episodic_metrics`, the other rollout-
buffer utilities it belongs with)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from navix.agents.models import MLPEncoder, TransformerBlock, TransformerEncoder


HIDDEN_SIZE = 8
CONTEXT = 4


def test_transformer_block_preserves_shape():
    block = TransformerBlock(hidden_size=HIDDEN_SIZE, num_heads=2)
    x = jax.random.normal(jax.random.PRNGKey(0), (CONTEXT, HIDDEN_SIZE))
    params = block.init(jax.random.PRNGKey(1), x)
    y = block.apply(params, x)
    assert y.shape == x.shape


def test_transformer_encoder_output_shape():
    encoder = TransformerEncoder(
        frame_encoder=MLPEncoder(hidden_size=HIDDEN_SIZE),
        hidden_size=HIDDEN_SIZE,
        num_heads=2,
        num_layers=2,
        context=CONTEXT,
    )
    x = jax.random.normal(jax.random.PRNGKey(0), (CONTEXT, 5))  # 5 raw features/frame
    params = encoder.init(jax.random.PRNGKey(1), x)
    y = encoder.apply(params, x)
    assert y.shape == (HIDDEN_SIZE,)


def test_transformer_encoder_shares_frame_encoder_weights():
    # frame_encoder is called `context` times inside one TransformerEncoder
    # - it must be one shared parameter set, not `context` independent
    # copies (a bug here would silently multiply parameter count by
    # `context` and defeat the whole point of a *shared* per-frame
    # encoder).
    encoder = TransformerEncoder(
        frame_encoder=MLPEncoder(hidden_size=HIDDEN_SIZE),
        hidden_size=HIDDEN_SIZE,
        context=CONTEXT,
    )
    x = jax.random.normal(jax.random.PRNGKey(0), (CONTEXT, 5))
    params = encoder.init(jax.random.PRNGKey(1), x)

    frame_encoder_params = params["params"]["frame_encoder"]
    leaf_count = len(jax.tree_util.tree_leaves(frame_encoder_params))
    # exactly one MLPEncoder's worth of leaves (2 Dense layers -> 4
    # leaves: kernel+bias each), regardless of `context`.
    assert leaf_count == 4


def test_transformer_encoder_attends_across_context_not_just_current_frame():
    # changing an *earlier* frame in the window (not the last one) must
    # still change the output - otherwise this degenerates to a
    # single-frame encoder that ignores history entirely, defeating
    # issue #169's whole point.
    encoder = TransformerEncoder(
        frame_encoder=MLPEncoder(hidden_size=HIDDEN_SIZE),
        hidden_size=HIDDEN_SIZE,
        context=CONTEXT,
    )
    x = jax.random.normal(jax.random.PRNGKey(0), (CONTEXT, 5))
    params = encoder.init(jax.random.PRNGKey(1), x)

    y1 = encoder.apply(params, x)
    x_perturbed = x.at[0].set(x[0] + 10.0)  # only the oldest frame changes
    y2 = encoder.apply(params, x_perturbed)

    assert not jnp.allclose(y1, y2)


def test_transformer_encoder_jit_vmap_compatible():
    encoder = TransformerEncoder(
        frame_encoder=MLPEncoder(hidden_size=HIDDEN_SIZE),
        hidden_size=HIDDEN_SIZE,
        context=CONTEXT,
    )
    x = jax.random.normal(jax.random.PRNGKey(0), (3, CONTEXT, 5))  # (batch, context, ...)
    params = encoder.init(jax.random.PRNGKey(1), x[0])

    apply_fn = jax.jit(jax.vmap(encoder.apply, in_axes=(None, 0)))
    y = apply_fn(params, x)
    assert y.shape == (3, HIDDEN_SIZE)
