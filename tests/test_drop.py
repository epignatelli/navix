"""Drop needs empty floor, not the weaker movement notion of walkability."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import navix as nx
from navix.components import DISCARD_PILE_COORDS, EMPTY_POCKET_ID
from navix.entities import Entities
from navix.states import State


def make_state(carried="key", destination="floor"):
    grid = jnp.zeros((5, 5), dtype=jnp.int32)
    target = jnp.array([2, 2])
    entities = {
        Entities.PLAYER: nx.entities.Player(
            position=jnp.array([2, 1]),
            direction=jnp.array(0),
            pocket=jnp.array({"key": 1, "ball": 2, "box": 3, None: -1}[carried]),
        )[None],
        Entities.KEY: nx.entities.Key.create(
            position=jnp.stack(
                [
                    DISCARD_PILE_COORDS,
                    target if destination == "key" else jnp.array([1, 1]),
                ]
            ),
            colour=jnp.array([0, 1]),
            id=jnp.array([1, 4]),
        ),
        Entities.BALL: nx.entities.Ball.create(
            position=DISCARD_PILE_COORDS[None],
            colour=jnp.array([0]),
            probability=jnp.array([1.0]),
            id=jnp.array([2]),
        ),
        Entities.BOX: nx.entities.Box.create(
            position=DISCARD_PILE_COORDS[None],
            colour=jnp.array([0]),
            id=jnp.array([3]),
            pocket=jnp.array([EMPTY_POCKET_ID]),
        ),
        Entities.GOAL: nx.entities.Goal.create(
            position=(target if destination == "goal" else jnp.array([3, 3]))[None],
            probability=jnp.array([1.0]),
        ),
        Entities.DOOR: nx.entities.Door.create(
            position=(target if "door" in destination else jnp.array([1, 3]))[None],
            colour=jnp.array([0]),
            open=jnp.array([destination == "open_door"]),
            requires=jnp.array([1 if destination == "locked_door" else -1]),
        ),
    }
    if destination in {"ball", "box"}:
        entity = entities[destination]
        blocker = entity.replace(position=target[None], id=jnp.array([4]))
        entities[destination] = jax.tree_util.tree_map(
            lambda a, b: jnp.concatenate([a, b]), entity, blocker
        )
    if destination == "wall":
        grid = grid.at[tuple(target)].set(-1)
    if destination == "lava":
        entities[Entities.LAVA] = nx.entities.Lava.create(position=target[None])
    return State(
        key=jax.random.PRNGKey(0),
        grid=grid,
        entities=entities,
        cache=nx.rendering.cache.RenderingCache.init(grid),
    )


def assert_same_tree(actual, expected):
    for a, b in zip(
        jax.tree_util.tree_leaves(actual),
        jax.tree_util.tree_leaves(expected),
        strict=True,
    ):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("carried", ["key", "ball", "box", None])
@pytest.mark.parametrize(
    "destination",
    [
        "wall",
        "key",
        "ball",
        "box",
        "open_door",
        "closed_door",
        "locked_door",
        "goal",
        "lava",
    ],
)
def test_drop_on_occupied_cell_is_noop(carried, destination):
    state = make_state(carried, destination)
    # Include events: attempting to drop must not impersonate walking into a
    # goal/hazard, or charge a wall-hit cost while the player stands still.
    assert_same_tree(jax.jit(nx.actions.drop)(state), state)


@pytest.mark.parametrize("carried", ["key", "ball", "box"])
def test_drop_moves_only_the_carried_item_and_can_be_picked_up(carried):
    state = make_state(carried)
    after = jax.jit(nx.actions.drop)(state)
    assert int(after.get_player().pocket) == int(EMPTY_POCKET_ID)
    np.testing.assert_array_equal(
        after.get_player().position, state.get_player().position
    )
    for name in (Entities.KEY, Entities.BALL, Entities.BOX):
        expected = state.entities[name].position
        if name == carried:
            expected = expected.at[0].set(jnp.array([2, 2]))
        np.testing.assert_array_equal(after.entities[name].position, expected)
    assert_same_tree(after.events, state.events)
    repicked = jax.jit(nx.actions.pickup)(after)
    assert_same_tree(repicked.entities, state.entities)


def test_drop_with_empty_pocket_on_floor_is_noop():
    state = make_state(None)
    assert_same_tree(jax.jit(nx.actions.drop)(state), state)


@pytest.mark.parametrize("direction", [0, 1, 2, 3])
def test_drop_outside_grid_is_noop(direction):
    state = make_state()
    position = jnp.array([[2, 4], [4, 2], [2, 0], [0, 2]])[direction]
    state = state.set_player(
        state.get_player().replace(position=position, direction=jnp.array(direction))
    )
    assert_same_tree(jax.jit(nx.actions.drop)(state), state)


@pytest.mark.parametrize("carried", ["key", None])
def test_environment_drop_facing_goal_does_not_reward_or_terminate(carried):
    env = nx.make("Navix-DoorKey-5x5-v0", penalty_coeff=0.0)
    timestep = env.reset(jax.random.PRNGKey(0))
    state = timestep.state
    state = state.set_player(
        state.get_player().replace(
            position=jnp.array([2, 1]),
            direction=jnp.array(0),
            pocket=state.get_keys().id[0] if carried else EMPTY_POCKET_ID,
        )
    )
    state = state.set_keys(state.get_keys().replace(position=DISCARD_PILE_COORDS[None]))
    state = state.set_goals(state.get_goals().replace(position=jnp.array([[2, 2]])))
    state = state.set_doors(state.get_doors().replace(position=jnp.array([[1, 3]])))
    state = state.set_walls(
        state.get_walls().replace(
            position=jnp.broadcast_to(
                DISCARD_PILE_COORDS, state.get_walls().position.shape
            ),
        )
    )
    timestep = timestep.replace(state=state.replace(grid=jnp.zeros_like(state.grid)))
    result = jax.jit(env.step)(
        timestep, jnp.array(env.action_set.index(nx.actions.drop))
    )
    assert float(result.reward) == 0
    assert not bool(result.is_done())
    assert_same_tree(result.state.entities, timestep.state.entities)


def test_vmap_can_drop_and_reject_occupied_cells_in_the_same_batch():
    states = jax.tree_util.tree_map(
        lambda *leaves: jnp.stack(leaves),
        make_state("key", "floor"),
        make_state("key", "open_door"),
        make_state(None, "goal"),
    )
    result = jax.jit(jax.vmap(nx.actions.drop))(states)
    np.testing.assert_array_equal(
        result.entities[Entities.PLAYER].pocket[:, 0], [-1, 1, -1]
    )
    np.testing.assert_array_equal(result.entities[Entities.KEY].position[0, 0], [2, 2])
    for index in (1, 2):
        assert_same_tree(
            jax.tree_util.tree_map(lambda leaf: leaf[index], result),
            jax.tree_util.tree_map(lambda leaf: leaf[index], states),
        )
