import jax.numpy as jnp
from navix.entities import Door


def test_open_unlocked():
    # open=1, requires=-1 (unlocked)
    door = Door(
        position=jnp.zeros((1, 2)),
        requires=jnp.asarray([-1]),
        open=jnp.asarray([1]),
        colour=jnp.asarray([0], dtype=jnp.uint8),
    )
    state = door.symbolic_state
    assert jnp.all(state == 0), "Expected state to be 0 for open unlocked door."


def test_closed_unlocked():
    # open=0, requires=-1 (unlocked)
    door = Door(
        position=jnp.zeros((1, 2)),
        requires=jnp.asarray([-1]),
        open=jnp.asarray([0]),
        colour=jnp.asarray([0], dtype=jnp.uint8),
    )
    state = door.symbolic_state
    assert jnp.all(state == 1), "Expected state to be 1 for closed unlocked door."


def test_closed_locked():
    # open=0, requires=5 (locked)
    door = Door(
        position=jnp.zeros((1, 2)),
        requires=jnp.asarray([5]),
        open=jnp.asarray([0]),
        colour=jnp.asarray([0], dtype=jnp.uint8),
    )
    state = door.symbolic_state
    assert jnp.all(state == 2), "Expected state to be 2 for closed locked door."


def test_open_locked():
    # open=1, requires=5 (locked, but open)
    door = Door(
        position=jnp.zeros((1, 2)),
        requires=jnp.asarray([5]),
        open=jnp.asarray([1]),
        colour=jnp.asarray([0], dtype=jnp.uint8),
    )
    state = door.symbolic_state
    assert jnp.all(state == 0), "Expected state to be 0 for open locked door."
