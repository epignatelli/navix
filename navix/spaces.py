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


from __future__ import annotations
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import Array
from flax import struct

Shape = Tuple[int, ...]


class Space(struct.PyTreeNode):
    shape: Shape = struct.field(pytree_node=False)
    dtype: jnp.dtype = struct.field(pytree_node=False)
    minimum: Array
    maximum: Array

    def sample(self, key: Array) -> Array:
        raise NotImplementedError()


class Discrete(Space):
    @classmethod
    def create(
        cls, n_elements: int | jax.Array, shape: Shape = (), dtype=jnp.int32
    ) -> Discrete:
        return Discrete(
            shape=shape,
            dtype=dtype,
            minimum=jnp.asarray(0),
            maximum=jnp.asarray(n_elements) - 1,
        )

    def sample(self, key: Array) -> Array:
        item = jax.random.randint(key, self.shape, self.minimum, self.maximum)
        # randint cannot draw jnp.uint, so we cast it later
        return jnp.asarray(item, dtype=self.dtype)


class Continuous(Space):
    @classmethod
    def create(
        cls, shape: Shape, minimum: Array, maximum: Array, dtype=jnp.float32
    ) -> Continuous:
        return Continuous(shape=shape, dtype=dtype, minimum=minimum, maximum=maximum)

    def sample(self, key: Array) -> Array:
        assert jnp.issubdtype(self.dtype, jnp.floating)
        # see: https://github.com/google/jax/issues/14003
        lower = jnp.nan_to_num(self.minimum)
        upper = jnp.nan_to_num(self.maximum)
        return jax.random.uniform(
            key, self.shape, minval=lower, maxval=upper, dtype=self.dtype
        )
