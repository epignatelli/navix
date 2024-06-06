import sys

import jax
import jax.numpy as jnp
from navix.spaces import Continuous, Discrete


MAX_INT = 100_000_000
MIN_INT = -100_000_000


def test_discrete():
    key = jax.random.PRNGKey(42)
    elements = (5, 0, MAX_INT, MIN_INT)
    shapes = ((), (0,), (0, 0), (1, 2), (5, 5))
    dtypes = (jnp.int8, jnp.int16, jnp.int32)
    for element in elements:
        for shape in shapes:
            for dtype in dtypes:
                space = Discrete.create(element, shape, dtype)
                sample = space.sample(key)
                print(sample)
                assert jnp.all(jnp.logical_not(jnp.isnan(sample)))


def test_continuous():
    key = jax.random.PRNGKey(42)
    shapes = ((), (0,), (0, 0), (1, 2), (5, 5))
    min_max = [
        (0.0, 1.0),
        (0.0, 1),
        (0, 1),
        (1.0, -1.0),
        (MIN_INT, MAX_INT),
    ]
    for shape in shapes:
        for minimum, maximum in min_max:
            space = Continuous.create(
                shape=shape, minimum=jnp.asarray(minimum), maximum=jnp.asarray(maximum)
            )
            sample = space.sample(key)
            print(sample)
            assert jnp.all(jnp.logical_not(jnp.isnan(sample)))


if __name__ == "__main__":
    test_discrete()
    test_continuous()
