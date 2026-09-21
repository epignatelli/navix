"""First-person observations against MiniGrid's own, cell for cell.

Every state of a random walk is copied into a MiniGrid `Grid` - walls, doors,
keys, balls, boxes, goals, lava, the agent's pose and what it carries - and
MiniGrid's `gen_obs` image and `get_pov_render` frame are compared with
navix's, at several view sizes, along with the full `get_full_render` frame
`RGBImgObsWrapper` returns. MiniGrid indexes its encoded image `[x, y]`
and navix `[row, col]`, so MiniGrid's is transposed first.
"""

import jax
import numpy as np
import pytest
from minigrid.core.constants import IDX_TO_COLOR, OBJECT_TO_IDX
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Box, Door, Goal, Key, Lava, Wall
from minigrid.minigrid_env import MiniGridEnv

import navix as nx
from navix.entities import Entities, EntityIds
from navix.rendering.registry import TILE_SIZE

ENV_IDS = [
    "Navix-DoorKey-5x5-v0",
    "Navix-DoorKey-8x8-v0",
    "Navix-DoorKey-Random-16x16-v0",
    "Navix-KeyCorridorS3R1-v0",
    "Navix-KeyCorridorS4R3-v0",
    "Navix-KeyCorridorS6R3-v0",
    "Navix-FourRooms-v0",
    "Navix-SimpleCrossingS11N5-v0",
    "Navix-LavaGapS7-v0",
    "Navix-UnlockPickup-v0",
    "Navix-ObstructedMaze-1Dlhb-v0",
]
RADII = [1, 3, 5]
NUM_STEPS = 50


@pytest.fixture(autouse=True)
def restore_radius():
    # set_radius is module-global; leave it as found for the tests after
    radius = nx.observations.RADIUS
    yield
    nx.observations.set_radius(radius)


class BlankEnv(MiniGridEnv):
    """A MiniGrid env whose grid and agent are overwritten per state."""

    def __init__(self, width, height):
        super().__init__(
            mission_space=MissionSpace(mission_func=lambda: ""),
            width=width,
            height=height,
            max_steps=1,
            see_through_walls=False,
            agent_view_size=2 * nx.observations.RADIUS + 1,
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.agent_pos, self.agent_dir = (1, 1), 0


def instances(state, name):
    if name not in state.entities:
        return []
    entity = state.entities[name]
    return [
        jax.tree_util.tree_map(lambda x, i=i: x[i], entity)
        for i in range(entity.shape[0])
    ]


def to_minigrid(state):
    H, W = state.grid.shape
    grid = Grid(W, H)
    for r, c in np.argwhere(np.asarray(state.grid) == -1):
        grid.set(int(c), int(r), Wall())

    player = state.get_player()
    carrying = None
    builders = {
        Entities.WALL: lambda e: Wall(),
        Entities.LAVA: lambda e: Lava(),
        Entities.GOAL: lambda e: Goal(),
        Entities.DOOR: lambda e: Door(
            IDX_TO_COLOR[int(e.colour)],
            is_open=bool(e.open),
            is_locked=bool(e.locked),
        ),
        Entities.KEY: lambda e: Key(IDX_TO_COLOR[int(e.colour)]),
        Entities.BALL: lambda e: Ball(IDX_TO_COLOR[int(e.colour)]),
        Entities.BOX: lambda e: Box(IDX_TO_COLOR[int(e.colour)]),
    }
    for name, build in builders.items():
        for entity in instances(state, name):
            r, c = (int(x) for x in entity.position)
            if 0 <= r < H and 0 <= c < W:
                grid.set(c, r, build(entity))
            elif hasattr(entity, "id") and int(entity.id) == int(player.pocket):
                carrying = build(entity)

    env = BlankEnv(W, H)
    env.reset(seed=0)
    env.grid = grid
    r, c = (int(x) for x in player.position)
    env.agent_pos = (c, r)
    env.agent_dir = int(player.direction)
    env.carrying = carrying
    return env


def random_walk(env_id):
    env = nx.make(env_id, observation_fn=nx.observations.none)
    key = jax.random.PRNGKey(0)
    timestep = env.reset(key)
    step = jax.jit(env.step)
    states = [timestep.state]
    for action in jax.random.randint(key, (NUM_STEPS,), 0, len(env.action_set)):
        timestep = step(timestep, action)
        states.append(timestep.state)
    return states


@pytest.mark.parametrize("radius", RADII)
@pytest.mark.parametrize("env_id", ENV_IDS)
def test_symbolic_first_person_matches_minigrid(env_id, radius):
    nx.observations.set_radius(radius)
    # a fresh closure per radius, so jit re-traces with the new RADIUS
    observe = jax.jit(lambda state: nx.observations.symbolic_first_person(state))
    for state in random_walk(env_id):
        expected = to_minigrid(state).gen_obs()["image"].transpose(1, 0, 2)
        np.testing.assert_array_equal(np.asarray(observe(state)), expected)


@pytest.mark.parametrize("radius", RADII)
@pytest.mark.parametrize("env_id", ENV_IDS)
def test_categorical_first_person_matches_minigrid(env_id, radius):
    # categorical_first_person is MiniGrid's object channel with two
    # exceptions its docstring gives: empty cells read 0, and the player's
    # own cell reads PLAYER while the pocket is empty.
    nx.observations.set_radius(radius)
    observe = jax.jit(lambda state: nx.observations.categorical_first_person(state))
    own = (2 * radius, radius)
    for state in random_walk(env_id):
        image = to_minigrid(state).gen_obs()["image"].transpose(1, 0, 2)
        expected = np.where(image[..., 0] == OBJECT_TO_IDX["empty"], 0, image[..., 0])
        if expected[own] == 0:
            expected[own] = int(EntityIds.PLAYER)
        np.testing.assert_array_equal(np.asarray(observe(state)), expected)


@pytest.mark.parametrize("radius", RADII)
@pytest.mark.parametrize("env_id", ENV_IDS)
def test_rgb_first_person_matches_minigrid(env_id, radius):
    # MiniGrid's RGBImgPartialObsWrapper renders get_pov_render at 8 pixels
    # a tile, navix's TILE_SIZE
    nx.observations.set_radius(radius)
    observe = jax.jit(lambda state: nx.observations.rgb_first_person(state))
    for state in random_walk(env_id):
        expected = to_minigrid(state).get_pov_render(tile_size=TILE_SIZE)
        np.testing.assert_array_equal(np.asarray(observe(state)), expected)


@pytest.mark.parametrize("radius", RADII)
@pytest.mark.parametrize("env_id", ENV_IDS)
def test_rgb_matches_minigrid(env_id, radius):
    # RGBImgObsWrapper renders get_frame with the env's default highlight,
    # which is the agent's field of view
    nx.observations.set_radius(radius)
    observe = jax.jit(lambda state: nx.observations.rgb(state))
    for state in random_walk(env_id):
        expected = to_minigrid(state).get_full_render(True, TILE_SIZE)
        np.testing.assert_array_equal(np.asarray(observe(state)), expected)
