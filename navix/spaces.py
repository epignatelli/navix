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
reward arrays.

A space says three things about an array that flows through an
environment - its `shape`, its `dtype`, and its element-wise
`[minimum, maximum]` bounds - and can `sample` a conforming array.

Each concrete space *is* a `distrax` distribution: `Discrete` subclasses
`distrax.Categorical` (uniform over `0 .. n_elements - 1`) and
`Continuous` subclasses `distrax.Uniform` (uniform over
`[minimum, maximum]`). So a space also carries `log_prob`, `entropy`,
`prob` and `mode` for free, and `isinstance(space, distrax.Distribution)`
holds. `Space` re-exports `distrax.Distribution` as the common supertype
navix annotates with.

The distribution is over a *single* element: the underlying
`distrax.Categorical.logits` is `1-D` (`(n_elements,)`), not
`(*shape, n_elements)`, so the descriptor stays cheap even for a large
image observation. `space.shape` / `space.dtype` are navix attributes on
top of that; `space.event_shape` / `space.batch_shape` (the `distrax`
introspection) are therefore `()`, not `space.shape`. `space.sample(key)`
takes the key positionally (navix convention) and returns a
`space.shape`-shaped array with iid elements - it does not follow
`distrax`'s keyword-only `sample(*, seed, sample_shape)` signature.
"""

from __future__ import annotations
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
import distrax

Shape = Tuple[int, ...]
"""An array shape, i.e. a tuple of ints (`()` for a scalar)."""

Space = distrax.Distribution
"""The common supertype of every navix space, i.e. `distrax.Distribution`.
An `Environment` exposes three spaces - `observation_space`,
`action_space` and `reward_space` - so a caller knows what `reset`/`step`
return and what `step` expects without running the environment."""


class Discrete(distrax.Categorical):
    """An integer-valued space: every element is one of the `n_elements`
    integers `0, 1, ..., n_elements - 1`, drawn uniformly and
    independently.

    Subclasses `distrax.Categorical` with uniform `1-D` logits of length
    `n_elements`, so `log_prob`/`entropy`/`prob`/`mode` all work (they
    broadcast over whatever shape you pass). `shape` describes the array
    the space stands for: `()` is a single categorical - the usual case
    for an action index (`Environment.action_space`) - while `(H, W)` /
    `(H, W, 3)` describe a grid or image observation whose every element
    is an independent categorical over the same `n_elements` values.

    Attributes:
        shape: shape of the array the space describes (`()` for a scalar).
        dtype: integer dtype of a sampled array (default `jnp.int32`).
        minimum: `0` (inclusive lower bound).
        maximum: `n_elements - 1` (inclusive upper bound).
        n: `n_elements`, i.e. `maximum + 1`.
    """

    def __init__(
        self,
        n_elements: int | Array,
        shape: Shape = (),
        dtype: jnp.dtype = jnp.int32,
    ):
        """Args:
        n_elements (int | Array): number of distinct values; `>= 1`.
        shape (tuple[int, ...]): shape of the integer array the space
            describes. `()` (the default) is a single scalar.
        dtype: integer dtype of the sampled array (default `jnp.int32`).
            Unsigned dtypes are allowed - `sample` draws with a signed
            generator and casts."""
        super().__init__(logits=jnp.zeros((int(n_elements),), dtype=jnp.float32))
        self.shape: Shape = tuple(int(d) for d in shape)
        self._np_dtype: np.dtype = jnp.dtype(dtype)

    @classmethod
    def create(
        cls, n_elements: int | Array, shape: Shape = (), dtype: jnp.dtype = jnp.int32
    ) -> Discrete:
        """Builds a `Discrete` space over `0 .. n_elements - 1`. Kept as a
        named constructor for parity with the rest of navix; identical to
        `Discrete(n_elements, shape, dtype)`.

        Args:
            n_elements (int | Array): number of distinct values; `>= 1`.
            shape (tuple[int, ...]): shape of the integer array (default
                `()`, a scalar).
            dtype: integer dtype of the sampled array (default
                `jnp.int32`).

        Returns:
            Discrete: the space, with `minimum = 0` and
            `maximum = n_elements - 1`."""
        return cls(n_elements, shape, dtype)

    def sample(self, key: Array) -> Array:  # type: ignore[override]
        """Draws integers uniformly from `0 .. n_elements - 1`,
        independently per element. Takes the key positionally (navix
        convention), not as `distrax`'s keyword-only `seed`.

        Args:
            key (Array): a `jax.random` PRNG key.

        Returns:
            Array: shape `shape`, dtype `dtype`."""
        # randint's `maxval` is exclusive, so `num_categories` (not
        # `num_categories - 1`) makes the top value reachable. randint
        # cannot target an unsigned dtype, so draw signed and cast.
        item = jax.random.randint(key, self.shape, 0, self.num_categories)
        return jnp.asarray(item, dtype=self._np_dtype)

    @property
    def dtype(self) -> np.dtype:  # type: ignore[override]
        """The dtype of a sampled array (a navix attribute; `distrax`
        would otherwise infer it from `sample`)."""
        return self._np_dtype

    @property
    def minimum(self) -> Array:
        """Element-wise lower bound (inclusive): `0`."""
        return jnp.asarray(0)

    @property
    def maximum(self) -> Array:
        """Element-wise upper bound (inclusive): `n_elements - 1`."""
        return jnp.asarray(self.num_categories - 1)

    @property
    def n(self) -> Array:
        """The number of distinct values, `n_elements` (`maximum + 1`).
        For an action space, `len(env.action_set)`."""
        return jnp.asarray(self.num_categories)

    def replace(self, **changes) -> Discrete:
        """Returns a copy with some of `n_elements` / `shape` / `dtype`
        overridden (mirrors `flax.struct`'s `replace`; used e.g. to flatten
        an observation space: `space.replace(shape=(prod(space.shape),))`)."""
        return Discrete(
            changes.get("n_elements", self.num_categories),
            changes.get("shape", self.shape),
            changes.get("dtype", self._np_dtype),
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Discrete)
            and int(self.num_categories) == int(other.num_categories)
            and self.shape == other.shape
            and self._np_dtype == other._np_dtype
        )

    def __hash__(self) -> int:
        return hash(
            ("Discrete", int(self.num_categories), self.shape, str(self._np_dtype))
        )

    def __repr__(self) -> str:
        return (
            f"Discrete(n_elements={int(self.num_categories)}, "
            f"shape={self.shape}, dtype={self._np_dtype.name})"
        )


class Continuous(distrax.Uniform):
    """A floating-point space: every element lies in `[minimum, maximum]`,
    drawn uniformly and independently.

    Subclasses `distrax.Uniform`, so `log_prob`/`entropy`/`prob` work.
    navix uses it for `reward_space` (`shape=()`, bounds `[-1, 1]` by
    default) and for float observations. Bounds may be infinite
    (`-jnp.inf` / `jnp.inf`) to express "unbounded"; `sample` maps those
    to the largest finite value of `dtype` (via `jnp.nan_to_num`) so an
    unbounded space still yields a number rather than `nan`.

    Attributes:
        shape: shape of the array the space describes (`()` for a scalar).
        dtype: floating dtype of a sampled array (default `jnp.float32`).
        minimum: element-wise lower bound (`low`), broadcast to `shape`.
        maximum: element-wise upper bound (`high`), broadcast to `shape`.
    """

    def __init__(
        self,
        shape: Shape,
        minimum: Array,
        maximum: Array,
        dtype: jnp.dtype = jnp.float32,
    ):
        """Args:
        shape (tuple[int, ...]): shape of the array (`()` is a scalar).
        minimum (Array): element-wise lower bound (inclusive); a scalar
            broadcasts to `shape`. May be `-jnp.inf`.
        maximum (Array): element-wise upper bound (inclusive); a scalar
            broadcasts to `shape`. May be `jnp.inf`.
        dtype: floating dtype of the sampled array (default
            `jnp.float32`)."""
        np_dtype = jnp.dtype(dtype)
        low = jnp.broadcast_to(jnp.asarray(minimum, np_dtype), shape)
        high = jnp.broadcast_to(jnp.asarray(maximum, np_dtype), shape)
        super().__init__(low=low, high=high)
        self.shape: Shape = tuple(int(d) for d in shape)
        self._np_dtype: np.dtype = np_dtype

    @classmethod
    def create(
        cls,
        shape: Shape,
        minimum: Array,
        maximum: Array,
        dtype: jnp.dtype = jnp.float32,
    ) -> Continuous:
        """Builds a `Continuous` space. Kept as a named constructor for
        parity with the rest of navix; identical to
        `Continuous(shape, minimum, maximum, dtype)`.

        Args:
            shape (tuple[int, ...]): shape of the array (`()` is a scalar).
            minimum (Array): element-wise lower bound; may be `-jnp.inf`.
            maximum (Array): element-wise upper bound; may be `jnp.inf`.
            dtype: floating dtype of the sampled array (default
                `jnp.float32`).

        Returns:
            Continuous: the space."""
        return cls(shape, minimum, maximum, dtype)

    def sample(self, key: Array) -> Array:  # type: ignore[override]
        """Draws values uniformly from `[minimum, maximum)`, independently
        per element. Takes the key positionally (navix convention), not as
        `distrax`'s keyword-only `seed`. Infinite bounds are first mapped
        to the largest finite value of `dtype` (via `jnp.nan_to_num`), so
        an unbounded space still yields a number rather than `nan`.

        Args:
            key (Array): a `jax.random` PRNG key.

        Returns:
            Array: shape `shape`, dtype `dtype`."""
        # see: https://github.com/google/jax/issues/14003
        lower = jnp.nan_to_num(self.low)
        upper = jnp.nan_to_num(self.high)
        return jax.random.uniform(
            key, self.shape, self._np_dtype, minval=lower, maxval=upper
        )

    @property
    def dtype(self) -> np.dtype:  # type: ignore[override]
        """The dtype of a sampled array (a navix attribute; `distrax`
        would otherwise infer it from `sample`)."""
        return self._np_dtype

    @property
    def minimum(self) -> Array:
        """Element-wise lower bound (inclusive), i.e. `distrax.Uniform.low`."""
        return self.low

    @property
    def maximum(self) -> Array:
        """Element-wise upper bound (inclusive), i.e. `distrax.Uniform.high`."""
        return self.high

    def replace(self, **changes) -> Continuous:
        """Returns a copy with some of `shape` / `minimum` / `maximum` /
        `dtype` overridden (mirrors `flax.struct`'s `replace`). When only
        `shape` changes, the current bounds are collapsed to their
        min/max scalar and re-broadcast to it (navix builds `Continuous`
        with constant bounds, so this is exact)."""
        low = changes.get("minimum", jnp.min(self.low) if self.low.size else self.low)
        high = changes.get(
            "maximum", jnp.max(self.high) if self.high.size else self.high
        )
        return Continuous(
            changes.get("shape", self.shape),
            low,
            high,
            changes.get("dtype", self._np_dtype),
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Continuous)
            and self.shape == other.shape
            and self._np_dtype == other._np_dtype
            and bool(jnp.all(self.low == other.low))
            and bool(jnp.all(self.high == other.high))
        )

    def __hash__(self) -> int:
        return hash(("Continuous", self.shape, str(self._np_dtype)))

    def __repr__(self) -> str:
        return (
            f"Continuous(shape={self.shape}, dtype={self._np_dtype.name}, "
            f"minimum={np.asarray(self.low).min()}, "
            f"maximum={np.asarray(self.high).max()})"
        )
