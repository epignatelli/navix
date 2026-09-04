import sys

import jax
import jax.numpy as jnp
import numpy as np
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


def test_discrete_sample_covers_the_full_range():
    # https://github.com/epignatelli/navix/issues/210 - `Discrete.sample`
    # passed the inclusive `maximum` as randint's exclusive `maxval`, so
    # the top value `n_elements - 1` was never drawn.
    n = 5
    space = Discrete.create(n)
    keys = jax.random.split(jax.random.PRNGKey(0), 2000)
    samples = np.asarray(jax.vmap(space.sample)(keys))
    assert samples.min() == 0
    assert samples.max() == n - 1  # the top value is reachable
    assert set(np.unique(samples).tolist()) == set(range(n))


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
    test_discrete_sample_covers_the_full_range()
    test_continuous()
