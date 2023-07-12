import jax
import jax.numpy as jnp

import navix as nx
from navix.entities import Goal, Player


def test_indexing():
    # batched entity with batch size 1
    entity = Player(
        position=jnp.ones((1, 2), dtype=jnp.int32),
        direction=jnp.ones((1,), jnp.int32),
        pocket=jnp.ones((1,), jnp.int32),
    )
    assert jnp.array_equal(entity[0].position, jnp.asarray((1, 1)))
    assert jnp.array_equal(entity[0].direction, jnp.asarray(1))


def test_get_sprites():
    # batched entity with batch size 1
    entity = Goal(position=jnp.ones((1, 2)), probability=jnp.ones((1,)))
    assert entity.sprite.shape == (1, nx.graphics.TILE_SIZE, nx.graphics.TILE_SIZE, 3)

    # batched entity with batch size > 1
    entity = Goal(position=jnp.ones((5, 2)), probability=jnp.ones((5,)))
    assert entity.sprite.shape == (5, nx.graphics.TILE_SIZE, nx.graphics.TILE_SIZE, 3)


if __name__ == "__main__":
    test_indexing()
    # test_get_sprites()
    jax.jit(test_get_sprites)()
