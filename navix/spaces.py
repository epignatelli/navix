# Copyright [2023] The Helx Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""`Space` descriptors for an environment's observation, action and
reward arrays - shape, dtype and element-wise bounds, plus a `sample`
that draws a conforming array. `Discrete` for integers, `Continuous` for
floats.
"""

from __future__ import annotations
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import Array
from flax import struct

Shape = Tuple[int, ...]
"""An array shape, i.e. a tuple of ints (`()` for a scalar)."""


class Space(struct.PyTreeNode):
    """Describes one array that flows through an environment: its shape, its
    dtype, and its element-wise value bounds.

    An [`Environment`][navix.environments.environment.Environment] exposes
    three of these - `observation_space`, `action_space` and
    `reward_space` - so a caller knows what `reset`/`step` will return and
    what `step` expects, without having to run the environment. `sample`
    draws a random array that conforms to the space, which is useful for
    smoke tests and for shaping neural-network inputs/outputs.

    Use the `create` classmethod of a concrete subclass
    ([`Discrete`][navix.spaces.Discrete] or
    [`Continuous`][navix.spaces.Continuous]) to build one; the bare
    constructor does no validation or bound broadcasting.

    Attributes:
        shape: the shape of the array the space describes. `()` means a
            single scalar (e.g. a discrete action index or a scalar
            reward); `(H, W)` / `(H, W, 3)` etc. describe a grid or image
            observation, with every element independently constrained to
            `[minimum, maximum]`.
        dtype: the array's dtype (e.g. `jnp.int32` for a `Discrete`
            action, `jnp.uint8` for a pixel observation, `jnp.float32`
            for a reward).
        minimum: element-wise lower bound (inclusive). A scalar array
            broadcasts to `shape`.
        maximum: element-wise upper bound (inclusive). A scalar array
            broadcasts to `shape`."""

    shape: Shape = struct.field(pytree_node=False)
    dtype: jnp.dtype = struct.field(pytree_node=False)
    minimum: Array
    maximum: Array

    def sample(self, key: Array) -> Array:
        """Draws one array of shape `shape` and dtype `dtype` whose
        elements lie within `[minimum, maximum]`.

        Args:
            key (Array): a `jax.random` PRNG key.

        Returns:
            Array: the sampled array, shape `shape`, dtype `dtype`.

        Raises:
            NotImplementedError: `Space` is abstract; call `sample` on a
                `Discrete` or `Continuous` instance."""
        raise NotImplementedError()


class Discrete(Space):
    """An integer-valued space: every element is one of the `n_elements`
    integers `0, 1, ..., n_elements - 1`.

    With `shape=()` it describes a single categorical value - the usual
    case for an action index (`Environment.action_space`). With a
    non-empty `shape` it describes an array of independent categoricals,
    e.g. a `categorical` observation is `Discrete` over entity tags with
    `shape=(H, W)`."""

    @classmethod
    def create(
        cls, n_elements: int | jax.Array, shape: Shape = (), dtype=jnp.int32
    ) -> Discrete:
        """Builds a `Discrete` space over `0 .. n_elements - 1`.

        Args:
            n_elements (int | Array): number of distinct values; must be
                `>= 1`. Stored as `maximum = n_elements - 1` (with
                `minimum = 0`), so `space.n` recovers it.
            shape (tuple[int, ...]): shape of the integer array the space
                describes. `()` (the default) is a single scalar.
            dtype: integer dtype of the sampled array (default
                `jnp.int32`). Unsigned dtypes are allowed - `sample`
                draws with a signed generator and casts.

        Returns:
            Discrete: the space, with `minimum = 0` and
            `maximum = n_elements - 1`."""
        return Discrete(
            shape=shape,
            dtype=dtype,
            minimum=jnp.asarray(0),
            maximum=jnp.asarray(n_elements) - 1,
        )

    def sample(self, key: Array) -> Array:
        """Draws integers uniformly from `0 .. n_elements - 1`,
        independently per element.

        Args:
            key (Array): a `jax.random` PRNG key.

        Returns:
            Array: shape `shape`, dtype `dtype`."""
        # `maximum` is the inclusive top value (`n_elements - 1`), but
        # `jax.random.randint`'s `maxval` is exclusive - pass `+ 1` so the
        # top value is actually reachable.
        item = jax.random.randint(key, self.shape, self.minimum, self.maximum + 1)
        # randint cannot draw jnp.uint, so we cast it later
        return jnp.asarray(item, dtype=self.dtype)

    @property
    def n(self) -> Array:
        """The number of distinct values, `n_elements` (i.e.
        `maximum + 1`). For an action space, `len(env.action_set)`."""
        return self.maximum + 1


class Continuous(Space):
    """A floating-point space: every element lies in `[minimum, maximum]`.

    navix uses it for `reward_space` (`shape=()`, bounds `[-1, 1]` by
    default) and for float observations. Bounds may be infinite
    (`-jnp.inf` / `jnp.inf`) to express "unbounded"; `sample` then falls
    back to a finite range (see below)."""

    @classmethod
    def create(
        cls, shape: Shape, minimum: Array, maximum: Array, dtype=jnp.float32
    ) -> Continuous:
        """Builds a `Continuous` space.

        Args:
            shape (tuple[int, ...]): shape of the array the space
                describes. `()` is a scalar (e.g. a reward).
            minimum (Array): element-wise lower bound (inclusive); a
                scalar broadcasts to `shape`. May be `-jnp.inf`.
            maximum (Array): element-wise upper bound (inclusive); a
                scalar broadcasts to `shape`. May be `jnp.inf`.
            dtype: floating dtype of the sampled array (default
                `jnp.float32`).

        Returns:
            Continuous: the space."""
        return Continuous(shape=shape, dtype=dtype, minimum=minimum, maximum=maximum)

    def sample(self, key: Array) -> Array:
        """Draws values uniformly from `[minimum, maximum)`, independently
        per element. Infinite bounds are first mapped to the largest
        finite value of `dtype` (via `jnp.nan_to_num`), so an unbounded
        space still yields a finite draw rather than `nan`.

        Args:
            key (Array): a `jax.random` PRNG key.

        Returns:
            Array: shape `shape`, dtype `dtype`.

        Raises:
            AssertionError: if `dtype` is not a floating type."""
        assert jnp.issubdtype(self.dtype, jnp.floating)
        # see: https://github.com/google/jax/issues/14003
        lower = jnp.nan_to_num(self.minimum)
        upper = jnp.nan_to_num(self.maximum)
        return jax.random.uniform(
            key, self.shape, minval=lower, maxval=upper, dtype=self.dtype
        )
