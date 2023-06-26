import jax
import jax.numpy as jnp

import navix as nx
from navix.entities import Goal


def test_get_sprites():
    # unbatched entity
    entity = Goal.create(position=jnp.asarray((1, 1)), probability=jnp.asarray(1.0))
    sprite = entity.get_sprite(nx.graphics.SPRITES_REGISTRY)
    assert sprite.shape == (nx.graphics.TILE_SIZE, nx.graphics.TILE_SIZE, 3)

    # batched entity with batch size 1
    entity = Goal.create(position=jnp.ones((1, 2)), probability=jnp.ones((1,)), tag=jnp.ones((1,)))
    sprite = entity.get_sprite(nx.graphics.SPRITES_REGISTRY)
    assert sprite.shape == (1, nx.graphics.TILE_SIZE, nx.graphics.TILE_SIZE, 3)

    # batched entity with batch size > 1
    entity = Goal.create(position=jnp.ones((5, 2)), probability=jnp.ones((5,)), tag=jnp.ones((5,)))
    sprite = entity.get_sprite(nx.graphics.SPRITES_REGISTRY)
    assert sprite.shape == (5, nx.graphics.TILE_SIZE, nx.graphics.TILE_SIZE, 3)


if __name__ == "__main__":
    test_get_sprites()
    jax.jit(test_get_sprites)()
