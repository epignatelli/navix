import jax
import jax.numpy as jnp
import numpy as np
import distrax

import navix as nx
from navix.spaces import Continuous, Discrete


def test_discrete():
    key = jax.random.PRNGKey(42)
    # n_elements is materialised as `distrax.Categorical` logits of that
    # length, so it is a small count in practice (action counts, entity
    # tags, 256 for a pixel) - not the ~1e8 the old hand-rolled space
    # tolerated.
    elements = (1, 2, 7, 11, 256)
    shapes = ((), (0,), (0, 0), (1, 2), (5, 5))
    for element in elements:
        for shape in shapes:
            # int dtype wide enough to hold `element - 1`
            dtype = jnp.int16 if element > 128 else jnp.int8
            space = Discrete.create(element, shape, dtype)
            sample = space.sample(key)
            assert sample.shape == shape
            assert sample.dtype == jnp.dtype(dtype)
            assert jnp.all((sample >= 0) & (sample < element))


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


def test_discrete_bounds_and_n():
    space = Discrete.create(7, shape=(3, 3), dtype=jnp.uint8)
    assert int(space.minimum) == 0
    assert int(space.maximum) == 6
    assert int(space.n) == 7
    assert space.shape == (3, 3)
    assert space.dtype == jnp.dtype(jnp.uint8)


def test_continuous():
    key = jax.random.PRNGKey(42)
    shapes = ((), (0,), (0, 0), (1, 2), (5, 5))
    min_max = [
        (0.0, 1.0),
        (0.0, 1),
        (0, 1),
        (1.0, -1.0),
        (-1e8, 1e8),
    ]
    for shape in shapes:
        for minimum, maximum in min_max:
            space = Continuous.create(
                shape=shape, minimum=jnp.asarray(minimum), maximum=jnp.asarray(maximum)
            )
            sample = space.sample(key)
            assert sample.shape == shape
            assert jnp.all(jnp.logical_not(jnp.isnan(sample)))


def test_continuous_infinite_bounds_map_to_finite():
    # `sample` runs the bounds through `jnp.nan_to_num`, so +/-inf bounds
    # become the largest finite float rather than producing `nan`.
    space = Continuous.create(
        shape=(4,), minimum=jnp.asarray(-jnp.inf), maximum=jnp.asarray(jnp.inf)
    )
    sample = space.sample(jax.random.PRNGKey(0))
    assert jnp.all(jnp.logical_not(jnp.isnan(sample)))


def test_spaces_are_distrax_distributions():
    # the "full replace": a space *is* a distrax distribution, so the
    # distribution API is available on top of the shape/dtype/bounds
    # descriptor.
    d = Discrete.create(4, shape=(2,))
    assert isinstance(d, distrax.Distribution)
    assert isinstance(d, distrax.Categorical)
    # uniform over 4 categories: log_prob is -log(4) everywhere
    np.testing.assert_allclose(float(d.log_prob(jnp.asarray(0))), -np.log(4), rtol=1e-6)
    np.testing.assert_allclose(float(d.entropy()), np.log(4), rtol=1e-6)

    c = Continuous.create(shape=(), minimum=jnp.asarray(0.0), maximum=jnp.asarray(2.0))
    assert isinstance(c, distrax.Distribution)
    assert isinstance(c, distrax.Uniform)
    np.testing.assert_allclose(float(c.log_prob(jnp.asarray(1.0))), -np.log(2), rtol=1e-6)


def test_space_replace_and_equality():
    d = Discrete.create(11, shape=(5, 5, 3), dtype=jnp.uint8)
    flat = d.replace(shape=(75,))
    assert flat.shape == (75,)
    assert int(flat.n) == 11 and flat.dtype == jnp.dtype(jnp.uint8)
    # value-based equality (two independently built spaces compare equal)
    assert d == Discrete.create(11, shape=(5, 5, 3), dtype=jnp.uint8)
    assert d != Discrete.create(11, shape=(5, 5, 3), dtype=jnp.int32)
    assert hash(d) == hash(Discrete.create(11, shape=(5, 5, 3), dtype=jnp.uint8))

    # Continuous.replace re-broadcasts constant bounds to the new shape
    c = Continuous.create(shape=(2, 3), minimum=jnp.asarray(-1.0), maximum=jnp.asarray(1.0))
    flat_c = c.replace(shape=(6,))
    assert flat_c.shape == (6,)
    assert flat_c.minimum.shape == (6,) and float(flat_c.minimum.min()) == -1.0
    assert flat_c == Continuous.create(
        shape=(6,), minimum=jnp.asarray(-1.0), maximum=jnp.asarray(1.0)
    )


def test_space_survives_jit_as_static_env_field():
    # Environment stores the spaces as pytree_node=False, so they land in
    # the jit treedef: they must be hashable and compare by value, and
    # `nx.make(id) == nx.make(id)` must hold (test_registry relies on it).
    a = nx.make("Navix-Empty-5x5-v0")
    b = nx.make("Navix-Empty-5x5-v0")
    assert a.observation_space == b.observation_space
    assert a.action_space == b.action_space
    assert a == b

    @jax.jit
    def rollout(env, key):
        ts = env.reset(key)
        return env.step(ts, env.action_space.sample(key)).reward

    r1 = rollout(a, jax.random.PRNGKey(0))
    r2 = rollout(b, jax.random.PRNGKey(0))  # same treedef -> cache hit, same result
    assert float(r1) == float(r2)


if __name__ == "__main__":
    test_discrete()
    test_discrete_sample_covers_the_full_range()
    test_continuous()
