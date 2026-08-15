import jax
import jax.numpy as jnp

from navix.entities import Door, Goal, Player
from navix.rendering.registry import TILE_SIZE


def test_door_open_derived_properties_are_bool():
    # Entity.walkable and Entity.transparent are both documented as boolean
    # arrays, but Door derives them from `open`, which Openable documents as
    # an integer 0/1.
    # However, callers disagree:
    #   - door_key.py passes a bool;
    #   - key_corridor.py and go_to_door.py pass ints.
    # And, observations.py:130 and :241 apply `~` to transparent, as bitwise
    # operation over an int dtype, it gives ~1 == -2 and ~0 == -1.
    # This becomes a problem when view_cone then multiplies its field with
    # it, because it results in unknown/invisible elements.
    for open_ in (jnp.asarray(0), jnp.asarray(False), jnp.asarray((0, 1))):
        door = Door(
            position=jnp.asarray([(1, 5)]),
            requires=jnp.asarray((0,)),
            open=open_,
            colour=jnp.zeros((1,), dtype=jnp.uint8),
        )
        assert door.transparent.dtype == jnp.bool_, open_.dtype
        assert door.walkable.dtype == jnp.bool_, open_.dtype


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
    entity = Goal.create(position=jnp.ones((1, 2)), probability=jnp.ones((1,)))
    assert entity.sprite.shape == (1, TILE_SIZE, TILE_SIZE, 3)

    # batched entity with batch size > 1
    entity = Goal.create(position=jnp.ones((5, 2)), probability=jnp.ones((5,)))
    assert entity.sprite.shape == (5, TILE_SIZE, TILE_SIZE, 3)


if __name__ == "__main__":
    test_door_open_derived_properties_are_bool()
    test_indexing()
    # test_get_sprites()
    jax.jit(test_get_sprites)()
