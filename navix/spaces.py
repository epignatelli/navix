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
    """Base class for all spaces in the game. Spaces define the shape and type of the \
        observations, actions and rewards in the game.
        The `sample` method is used to generate random samples from the space.

    !!! note
        To initialize a space, use the `create` method of the specific space class.
        
    TODO: 
        * maximum and minimum should be static objects, not arrays.
    But how do we handle the case when they are not scalars? Maybe numpy arrays?"""

    shape: Shape = struct.field(pytree_node=False)
    dtype: jnp.dtype = struct.field(pytree_node=False)
    minimum: Array
    maximum: Array

    def sample(self, key: Array) -> Array:
        """Generate a random sample from the space.

        Args:
            key (Array): A random key to generate the sample.

        Returns:
            Array: A random sample from the space."""
        raise NotImplementedError()


class Discrete(Space):
    @classmethod
    def create(
        cls, n_elements: int | jax.Array, shape: Shape = (), dtype=jnp.int32
    ) -> Discrete:
        """Create a discrete space with a given number of elements.

        Args:
            n_elements (int | jax.Array): The number of elements in the space.
            shape (Shape): The shape of the space.
            dtype (jnp.dtype): The data type of the space.

        Returns:
            Discrete: A discrete space with the given number of elements."""
        return Discrete(
            shape=shape,
            dtype=dtype,
            minimum=jnp.asarray(0),
            maximum=jnp.asarray(n_elements) - 1,
        )

    def sample(self, key: Array) -> Array:
        """Generate a random sample from the space.

        Args:
            key (Array): A random key to generate the sample.

        Returns:
            Array: A random sample from the space."""
        item = jax.random.randint(key, self.shape, self.minimum, self.maximum)
        # randint cannot draw jnp.uint, so we cast it later
        return jnp.asarray(item, dtype=self.dtype)

    @property
    def n(self) -> Array:
        """The number of elements in the space.

        Returns:
            Array: The number of elements in the space."""
        return self.maximum + 1


class Continuous(Space):
    @classmethod
    def create(
        cls, shape: Shape, minimum: Array, maximum: Array, dtype=jnp.float32
    ) -> Continuous:
        """Create a continuous space with a given shape, minimum and maximum values.

        Args:
            shape (Shape): The shape of the space.
            minimum (Array): The minimum value of the space.
            maximum (Array): The maximum value of the space.
            dtype (jnp.dtype): The data type of the space.

        Returns:
            Continuous: A continuous space with the given shape, minimum and maximum values.
        """
        return Continuous(shape=shape, dtype=dtype, minimum=minimum, maximum=maximum)

    def sample(self, key: Array) -> Array:
        """Generate a random sample from the space.

        Args:
            key (Array): A random key to generate the sample.

        Returns:
            Array: A random sample from the space."""
        assert jnp.issubdtype(self.dtype, jnp.floating)
        # see: https://github.com/google/jax/issues/14003
        lower = jnp.nan_to_num(self.minimum)
        upper = jnp.nan_to_num(self.maximum)
        return jax.random.uniform(
            key, self.shape, minval=lower, maxval=upper, dtype=self.dtype
        )
