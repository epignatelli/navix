# Copyright 2023 The Navix Authors.

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Thorough, hand-crafted-policy coverage of `EventsManager` (issue #139),
one real registered environment/seed per event type, verified against the
environment's actual map rather than a synthetic hand-built `State`
wherever a real environment can exercise the same code path.
`tests/test_issues.py::test_139` covers the exact original bug scenarios
(multi-ball collapse, cross-call-site clobbering) with a seed-swept
synthetic state; this file is the complementary "does every event
actually fire correctly, with the right position/colour, in real
gameplay" pass, plus direct coverage of the two `record_*` methods no
current action ever reaches (`record_door_unlock`, `record_ball_pickup`
- see their own tests below).

Every test reads the actually-placed entity positions from the live
reset state and derives its policy from those, rather than hardcoding
literal coordinates - `jax.random.PRNGKey(seed)` is only deterministic
*within* one jax version, not across them (confirmed the hard way: this
file originally hardcoded positions hand-read on jax 0.4.30, which CI's
jax 0.11.1 does not reproduce for the same seed - same environment code,
same seed, genuinely different procedurally-generated layout). Reading
positions from the live state instead of assuming them keeps these
tests meaningful regardless of which jax version produced the layout.

Every test calls `navix.actions`/`navix.transitions` functions directly
(not `env.step`), so movement is fully deterministic - `env.step`'s
`stochastic_transition` also runs `update_balls` (random per-step ball
movement) after every action, which would make navigation
non-reproducible."""

from __future__ import annotations

import jax
import jax.numpy as jnp

import navix as nx
from navix.components import EMPTY_POCKET_ID
from navix.entities import Entities
from navix.states import EventType, GRID


EAST, SOUTH, WEST, NORTH = 0, 1, 2, 3


def face(state, direction: int):
    """Rotates (0-3 times, whichever is fewer) until the player faces
    `direction`."""
    for _ in range(4):
        if int(state.get_player().direction) == direction:
            return state
        state = nx.actions.rotate_cw(state)
    raise AssertionError(f"could not face direction {direction}")


def walk(state, direction: int, steps: int):
    """Faces `direction`, then calls `actions.forward` `steps` times."""
    state = face(state, direction)
    for _ in range(steps):
        state = nx.actions.forward(state)
    return state


def walk_to(state, row: int, col: int):
    """Moves the player onto `(row, col)`, first aligning the row (N/S)
    then the column (E/W). Only used between cells hand-verified (per
    call site below) to have no obstacle along that L-shaped path - not
    a general pathfinder."""
    player = state.get_player()
    delta_row = row - int(player.position[0])
    if delta_row > 0:
        state = walk(state, SOUTH, delta_row)
    elif delta_row < 0:
        state = walk(state, NORTH, -delta_row)
    player = state.get_player()
    delta_col = col - int(player.position[1])
    if delta_col > 0:
        state = walk(state, EAST, delta_col)
    elif delta_col < 0:
        state = walk(state, WEST, -delta_col)
    return state


def test_grid_hit_boundary_not_wall_entity():
    # Navix-Empty-5x5-v0 is fully deterministic (no jax.random calls in
    # its _reset at all) and has no Wall entities - its boundary is pure
    # grid (grid[r, c] != 0), so walking into it must record (GRID,
    # HIT), not (WALL, HIT).
    env = nx.make("Navix-Empty-5x5-v0")
    timestep = env.reset(jax.random.PRNGKey(0))
    state = timestep.state
    assert Entities.WALL not in state.entities, (
        "test assumes Empty-5x5 has no Wall entities - if this changes, "
        "this test needs a different environment for a clean GRID-only case"
    )
    player = state.get_player()
    assert jnp.array_equal(player.position, jnp.asarray((1, 1)))
    assert int(player.direction) == 0  # east

    # face north then walk into the boundary one row above the player's
    # start - (0, 1), a plain grid cell.
    state = face(state, NORTH)
    state = nx.actions.forward(state)

    player = state.get_player()
    assert jnp.array_equal(player.position, jnp.asarray((1, 1))), (
        "player must not move into the (non-walkable) grid boundary"
    )
    assert state.events.happened((GRID, EventType.HIT)), (
        "expected the grid-boundary hit to be recorded under the GRID "
        "pseudo entity-type slot"
    )
    assert not state.events.happened((Entities.WALL, EventType.HIT)), (
        "there is no Wall entity here - (WALL, HIT) must not fire"
    )
    assert state.events.events[GRID, EventType.HIT].position.tolist() == [
        0,
        1,
    ], "the recorded grid-hit position should be the boundary cell that was hit"
    assert not state.events.happened((Entities.GOAL, EventType.REACH))

    # on_wall_hit must be True too - it ORs (GRID, HIT) and (WALL, HIT).
    assert bool(nx.events.on_wall_hit(state))


def test_wall_key_door_goal_sequence():
    # Navix-DoorKey-5x5-v0: player/goal positions are fixed (random_start
    # defaults to False), but door_row (and so which wall row has the
    # gap, and which row of the single-column first room the key lands
    # in) is genuinely randomised (navix/environments/door_key.py) - read
    # every position from the live state and derive the policy from
    # that. One coherent policy exercises WALL/HIT, KEY/PICKUP, DOOR/OPEN
    # and GOAL/REACH in narrative order.
    env = nx.make("Navix-DoorKey-5x5-v0")
    timestep = env.reset(jax.random.PRNGKey(0))
    state = timestep.state

    doors = state.get_doors()
    door_row, door_col = int(doors.position[0, 0]), int(doors.position[0, 1])
    assert bool(doors.requires[0] != -1), "the door must start locked"

    wall = state.entities[Entities.WALL]
    assert wall.position[:, 1].tolist() == [door_col] * wall.position.shape[0], (
        "every wall instance shares the door's own column"
    )
    wall_row = int(wall.position[0, 0])
    assert wall_row != door_row, (
        "the door's own row is excluded from the wall column by construction"
    )

    key = state.get_keys()
    key_row, key_col = int(key.position[0, 0]), int(key.position[0, 1])
    assert key_col == door_col - 1, "the key must be in the first room"

    goal = state.get_goals()
    goal_row, goal_col = int(goal.position[0, 0]), int(goal.position[0, 1])

    player = state.get_player()
    assert player.position.tolist() == [1, 1]
    assert int(player.direction) == 0  # east

    # 1. walk into a confirmed real Wall instance: blocked, WALL/HIT
    # fires for that instance only, GRID/HIT must NOT also fire (real
    # Wall entity, not a bare grid cell).
    state = walk_to(state, wall_row, door_col - 1)
    state = face(state, EAST)
    assert not state.events.happened((Entities.WALL, EventType.HIT))
    state = nx.actions.forward(state)
    assert state.get_player().position.tolist() == [wall_row, door_col - 1], (
        "must not move into a Wall"
    )
    wall_hit = state.events.events[Entities.WALL, EventType.HIT]
    assert bool(jnp.any(wall_hit.happened))
    hit_idx = int(jnp.argmax(wall_hit.happened))
    assert wall_hit.position[hit_idx].tolist() == [wall_row, door_col]
    assert not state.events.happened((GRID, EventType.HIT))
    assert bool(nx.events.on_wall_hit(state))

    # 2. pick up the key from the cell just before it (Key.walkable is
    # False, so pickup has to happen from an adjacent cell facing it,
    # not by stepping onto it).
    state = walk_to(state, key_row - 1, key_col)
    state = face(state, SOUTH)
    assert not state.events.happened((Entities.KEY, EventType.PICKUP))
    state = nx.actions.pickup(state)
    player = state.get_player()
    assert int(player.pocket) != int(EMPTY_POCKET_ID), (
        "the key's id should now be in the player's pocket"
    )
    key_pickup = state.events.events[Entities.KEY, EventType.PICKUP]
    assert bool(key_pickup.happened[0])
    assert key_pickup.position[0].tolist() == [key_row, key_col]
    assert int(key_pickup.colour[0]) == int(key.colour[0]), (
        "the recorded colour should be the actual picked-up key's own colour"
    )

    # 3. step onto the now-vacated key cell, then to the door.
    state = walk_to(state, key_row, key_col)
    state = walk_to(state, door_row, door_col - 1)
    state = face(state, EAST)

    # 4. open the door - the player's key matches, so it unlocks and
    # opens in the same action (see actions.open); DOOR/OPEN fires,
    # DOOR/UNLOCK does not (no current action ever calls record_door_
    # unlock - see test_door_unlock_and_ball_pickup_are_directly_
    # testable below for why, and direct coverage of it).
    assert not state.events.happened((Entities.DOOR, EventType.OPEN))
    state = nx.actions.open(state)
    doors_after = state.get_doors()
    assert bool(doors_after.open[0]), "the door should now be open"
    assert int(doors_after.requires[0]) == -1, "and unlocked"
    door_opening = state.events.events[Entities.DOOR, EventType.OPEN]
    assert bool(door_opening.happened[0])
    assert door_opening.position[0].tolist() == [door_row, door_col]
    assert not state.events.happened((Entities.DOOR, EventType.UNLOCK))

    # 5. walk through the now-open door onto the goal - GOAL/REACH fires.
    # (door_row can equal goal_row, in which case stepping through the
    # door already lands on the goal - so the "not yet reached" check
    # has to happen before any of these moves, not partway through.)
    assert not state.events.happened((Entities.GOAL, EventType.REACH))
    state = nx.actions.forward(state)  # onto the (now open) door cell
    state = walk_to(state, door_row, door_col + 1)  # into the second room
    state = walk_to(state, goal_row, goal_col)
    player = state.get_player()
    assert player.position.tolist() == [goal_row, goal_col]
    goal_reached = state.events.events[Entities.GOAL, EventType.REACH]
    assert bool(goal_reached.happened[0])
    assert goal_reached.position[0].tolist() == [goal_row, goal_col]
    assert bool(nx.events.on_goal_reached(state))

    # every earlier event recorded in this same episode must still be
    # True - EventsManager never resets happened back to False mid-
    # episode (see State.__post_init__/EventsManager.merge_event).
    assert bool(jnp.any(state.events.events[Entities.WALL, EventType.HIT].happened))
    assert bool(state.events.events[Entities.KEY, EventType.PICKUP].happened[0])
    assert bool(state.events.events[Entities.DOOR, EventType.OPEN].happened[0])


def test_lava_fall():
    # Navix-LavaGapS5-v0: player/goal positions are fixed, but the gap
    # row (and so which of the two lava-column rows actually holds lava)
    # is randomised (navix/environments/lava_gap.py) - read the live
    # positions rather than assuming them. Lava.walkable is True
    # (matching real MiniGrid semantics - stepping on lava ends the
    # episode via on_lava_fall termination rather than blocking
    # movement), so walking forward both moves the player onto it and
    # records LAVA/FALL, in the same step.
    env = nx.make("Navix-LavaGapS5-v0")
    timestep = env.reset(jax.random.PRNGKey(0))
    state = timestep.state

    lava = state.get_lavas()
    lava_row, lava_col = int(lava.position[0, 0]), int(lava.position[0, 1])
    player = state.get_player()
    assert player.position.tolist() == [1, 1]
    assert int(player.direction) == 0  # east
    assert lava_col > 1, "the lava column is always to the east of the player's start"

    state = walk_to(state, lava_row, lava_col - 1)
    state = face(state, EAST)
    assert not state.events.happened((Entities.LAVA, EventType.FALL))
    state = nx.actions.forward(state)

    player = state.get_player()
    assert player.position.tolist() == [lava_row, lava_col], (
        "the player should have moved onto the (walkable) lava tile"
    )
    lava_fall = state.events.events[Entities.LAVA, EventType.FALL]
    assert bool(jnp.any(lava_fall.happened))
    hit_idx = int(jnp.argmax(lava_fall.happened))
    assert lava_fall.position[hit_idx].tolist() == [lava_row, lava_col]
    assert bool(nx.events.on_lava_fall(state))


def test_ball_hit_walk_into():
    # Navix-Dynamic-Obstacles-5x5-v0: ball positions are placed fully at
    # random anywhere in the interior (navix/environments/
    # dynamic_obstacles.py's random_positions, excluding the player/goal
    # cells) - not confined to a predictable line the way DoorKey's key
    # or LavaGap's lava column are, so there's no small L-shaped walk
    # guaranteed obstacle-free between the player's start and an
    # arbitrary ball. Real reset (real player/goal/grid), balls
    # relocated to a fixed, hand-verifiable spot for the walk-into check
    # specifically - real Ball entities/events machinery throughout,
    # only the *position* is controlled. The genuinely-random-placement
    # case (and two-balls-hit-in-one-step) is covered by
    # tests/test_issues.py::test_139 instead, via a seed sweep rather
    # than by controlling positions directly.
    env = nx.make("Navix-Dynamic-Obstacles-5x5-v0")
    timestep = env.reset(jax.random.PRNGKey(0))
    state = timestep.state

    player = state.get_player()
    assert player.position.tolist() == [1, 1]
    assert int(player.direction) == 0  # east

    balls = state.get_balls()
    balls = balls.replace(position=jnp.asarray([[3, 2], [2, 2]]))
    state = state.replace(entities={**state.entities, Entities.BALL: balls})

    state = walk_to(state, 2, 1)
    state = face(state, EAST)

    assert not state.events.happened((Entities.BALL, EventType.HIT))
    state = nx.actions.forward(state)  # attempt (2, 1) -> (2, 2) = ball 1

    player = state.get_player()
    assert player.position.tolist() == [2, 1], (
        "must not move into a Ball (Ball.walkable is False)"
    )
    ball_hit = state.events.events[Entities.BALL, EventType.HIT]
    assert ball_hit.happened.tolist() == [False, True], (
        "only ball index 1 (2, 2) was walked into, not index 0 (3, 2)"
    )
    assert ball_hit.position[1].tolist() == [2, 2]
    assert int(ball_hit.colour[1]) == int(balls.colour[1])
    assert bool(nx.events.on_ball_hit(state))


def test_door_unlock_and_ball_pickup_are_directly_testable():
    # record_door_unlock and record_ball_pickup are the two record_*
    # methods no current navix action ever calls: actions.open only ever
    # calls record_door_opening (even when it unlocks a locked door in
    # the same step - see test_wall_key_door_goal_sequence above, where
    # DOOR/UNLOCK stays False despite the door being both unlocked and
    # opened), and actions.pickup only ever handles Key, never Ball
    # (confirmed via grep across navix/ before writing this test - no
    # call site for either exists outside states.py itself). That's a
    # pre-existing gap unrelated to issue #139 (not something this PR
    # changes), but merge_event's own correctness for these two slots
    # is still worth verifying directly, the same way the reachable
    # record_* methods are verified above through real gameplay.
    env = nx.make("Navix-DoorKey-5x5-v0")
    timestep = env.reset(jax.random.PRNGKey(0))
    state = timestep.state
    doors = state.get_doors()

    assert not state.events.happened((Entities.DOOR, EventType.UNLOCK))
    hit = jnp.asarray([True])
    events = state.events.record_door_unlock(doors, hit)
    door_unlock = events.events[Entities.DOOR, EventType.UNLOCK]
    assert bool(door_unlock.happened[0])
    assert door_unlock.position[0].tolist() == doors.position[0].tolist()
    assert int(door_unlock.colour[0]) == int(doors.colour[0])
    # the slot this didn't touch must be unaffected.
    assert not events.happened((Entities.DOOR, EventType.OPEN))

    env2 = nx.make("Navix-Dynamic-Obstacles-5x5-v0")
    timestep2 = env2.reset(jax.random.PRNGKey(0))
    state2 = timestep2.state
    balls = state2.get_balls()

    assert not state2.events.happened((Entities.BALL, EventType.PICKUP))
    hit2 = jnp.asarray([False, True])
    events2 = state2.events.record_ball_pickup(balls, hit2)
    ball_pickup = events2.events[Entities.BALL, EventType.PICKUP]
    assert ball_pickup.happened.tolist() == [False, True]
    assert ball_pickup.position[1].tolist() == balls.position[1].tolist()
    # BALL/HIT (a different slot for the same entity type) must be
    # unaffected by writing BALL/PICKUP.
    assert not events2.happened((Entities.BALL, EventType.HIT))


def test_happened_returns_false_not_keyerror_for_absent_entity_type():
    # Navix-Empty-5x5-v0 has no Wall/Key/Door/Lava/Ball entities at all -
    # EventsManager.create never allocates those slots for it, so
    # `happened` on any of them must gracefully return False, not raise.
    env = nx.make("Navix-Empty-5x5-v0")
    timestep = env.reset(jax.random.PRNGKey(0))
    state = timestep.state

    for entities in (Entities.WALL, Entities.KEY, Entities.DOOR, Entities.LAVA, Entities.BALL):
        assert entities not in state.entities

    assert not bool(state.events.happened((Entities.WALL, EventType.HIT)))
    assert not bool(state.events.happened((Entities.KEY, EventType.PICKUP)))
    assert not bool(state.events.happened((Entities.DOOR, EventType.OPEN)))
    assert not bool(state.events.happened((Entities.DOOR, EventType.UNLOCK)))
    assert not bool(state.events.happened((Entities.LAVA, EventType.FALL)))
    assert not bool(state.events.happened((Entities.BALL, EventType.HIT)))
    assert not bool(state.events.happened((Entities.BALL, EventType.PICKUP)))
    # on_wall_hit must still work (falls back to the always-present GRID
    # slot) even though this environment has no Wall entities.
    assert not bool(nx.events.on_wall_hit(state))
