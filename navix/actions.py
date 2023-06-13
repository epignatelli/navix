from __future__ import annotations


import jax
import jax.numpy as jnp

from .components import State
from .grid import mask_entity, remove_entity


DIRECTIONS = {0: "east", 1: "south", 2: "west", 3: "north"}


def _rotate(state: State, spin: int) -> State:
    player = state.entities["player/0"]
    direction = (player.direction + spin) % 3
    player = player.replace(direction=direction)
    state.entities["player/0"] = player
    return state


def _move(state: State, entity_id: int, direction: int) -> State:
    moves = (
        lambda x: jnp.roll(x, 1, axis=1),  # east
        lambda x: jnp.roll(x, 1, axis=0),  # south
        lambda x: jnp.roll(x, -1, axis=1),  # west
        lambda x: jnp.roll(x, -1, axis=0),  # north
    )
    mask = mask_entity(state.grid, entity_id)
    new_mask = jax.lax.switch(direction, moves, mask)

    grid = remove_entity(state.grid, entity_id)
    grid = jnp.where(new_mask, entity_id, grid)
    # TODO(epignatelli): handle collisions
    return state.replace(grid=grid)


def noop(state: State) -> State:
    return state


def rotate_cw(state: State) -> State:
    return _rotate(state, 1)


def rotate_ccw(state: State) -> State:
    return _rotate(state, -1)


def forward(state: State) -> State:
    entity_id = state.entities["player/0"].id
    direction = state.entities["player/0"].direction
    return _move(state, entity_id, direction)


def right(state: State) -> State:
    entity_id = state.entities["player/0"].id
    direction = state.entities["player/0"].direction + 1
    return _move(state, entity_id, direction)


def backward(state: State) -> State:
    entity_id = state.entities["player/0"].id
    direction = state.entities["player/0"].direction + 2
    return _move(state, entity_id, direction)


def left(state: State) -> State:
    entity_id = state.entities["player/0"].id
    direction = state.entities["player/0"].direction + 3
    return _move(state, entity_id, direction)


ACTIONS = {
    0: noop,
    1: rotate_cw,
    2: rotate_ccw,
    3: forward,
    4: right,
    5: backward,
    6: left,
}
