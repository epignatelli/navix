import jax
import jax.numpy as jnp

import navix as nx


def test_get_tiles():
    state = nx.components.State(
        key=jax.random.PRNGKey(0),
        grid=jnp.zeros((3, 3), dtype=jnp.int32),
        player=nx.components.Player(position=jnp.asarray((1, 1))),
        goals=nx.components.Goal(position=jnp.asarray((2, 2))[None]),
        keys=nx.components.Key(position=jnp.asarray((0, 0))[None]),
        doors=nx.components.Door(position=jnp.asarray((1, 2))[None]),
        cache=nx.graphics.RenderingCache.init(jnp.zeros((3, 3), dtype=jnp.int32)),
    )

    tiles_registry = nx.graphics.TILES_REGISTRY_WITH_DIRECTION

    tiles = state.get_tiles(tiles_registry=tiles_registry)