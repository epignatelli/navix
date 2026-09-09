"""Tests for MiniGrid-faithful first-person occlusion (`grid.process_vis`).

The golden windows below were produced by running MiniGrid's own
`minigrid.core.grid.Grid.process_vis` (v3.1.0) on the given transparency
map with the agent at the bottom-centre, and are pasted here verbatim so
the suite does not need MiniGrid installed to pin the behaviour.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from navix.grid import crop, process_vis


def parse(rows, true_char):
    return jnp.asarray([[c == true_char for c in row] for row in rows])


# (transparency, expected visibility), '#' opaque, 'V' visible.
# Row 0 is furthest ahead; the agent stands at the centre of the last row.
MINIGRID_CASES = (
    (
        (
            "..###..",
            ".#...##",
            "#.####.",
            "#.##...",
            "..#..#.",
            "#.###.#",
            "#.#....",
        ),
        (
            "-------",
            "-----VV",
            "---VVVV",
            "--VVVVV",
            "--VVVVV",
            "--VVVVV",
            "--VVVVV",
        ),
    ),
    (
        (
            ".#..#.#",
            "#.#.##.",
            ".##..##",
            "...##..",
            "..##.##",
            "#..####",
            "..#.#..",
        ),
        (
            "VVVVV--",
            "VVVVVV-",
            "VVVVVV-",
            "VVVV---",
            "VVVV---",
            "VVVVV--",
            "--VVV--",
        ),
    ),
    (
        (
            "...###.",
            "..#.##.",
            "...###.",
            "####...",
            "#.##..#",
            ".#...#.",
            ".....##",
        ),
        (
            "-----VV",
            "-----VV",
            "---VVVV",
            "VVVVVVV",
            "VVVVVVV",
            "VVVVVV-",
            "VVVVVV-",
        ),
    ),
    (
        (
            "..#.##.",
            ".#..#..",
            ".#.#...",
            "#.###..",
            ".#.##..",
            "#..##..",
            "#.#.#..",
        ),
        (
            "VVVVV--",
            "VVVVV--",
            "VVVV---",
            "VVVV---",
            "VVVV---",
            "VVVVV--",
            "--VVV--",
        ),
    ),
    (
        (
            "....##.",
            "..#..##",
            "..##...",
            "...#...",
            ".#.#...",
            "..##...",
            "..#.#.#",
        ),
        (
            "VVVVVV-",
            "--VVVVV",
            "---VVVV",
            "---VVVV",
            "---VVVV",
            "--VVVVV",
            "--VVV--",
        ),
    ),
    (
        (
            "#.#....",
            "..##...",
            "..#....",
            "...#..#",
            "...#..#",
            ".###...",
            "#.#....",
        ),
        (
            "--VVVVV",
            "--VVVVV",
            "--VVVVV",
            "---VVVV",
            "---VVVV",
            "--VVVVV",
            "--VVVVV",
        ),
    ),
    (
        (
            "..#....",
            "..#.#..",
            "#.#...#",
            ".##....",
            "..##...",
            "..#.#.#",
            ".##.###",
        ),
        (
            "--VVVVV",
            "--VVVVV",
            "--VVVVV",
            "--VVVVV",
            "--VVVVV",
            "--VVV--",
            "--VVV--",
        ),
    ),
    (
        (
            "....#..",
            ".#.#...",
            "#..##..",
            "....#.#",
            "...#.#.",
            "...##.#",
            "....#.#",
        ),
        (
            "VVVVV--",
            "VVVV---",
            "VVVVV--",
            "VVVVV--",
            "VVVV---",
            "VVVVV--",
            "VVVVV--",
        ),
    ),
)


def test_process_vis_matches_minigrid():
    for transparency, expected in MINIGRID_CASES:
        got = process_vis(parse(transparency, "."))
        np.testing.assert_array_equal(np.asarray(got), np.asarray(parse(expected, "V")))


def test_process_vis_flows_through_a_gap_in_a_wall():
    # Taken from Navix-FourRooms-v0 (seed 0, agent at row 8 col 6 facing
    # south), where the current view_cone disagrees with MiniGrid on 23 of
    # the 49 cells. MiniGrid propagates sideways without limit within a
    # row, so a single gap at the far edge of the wall lights everything
    # past it - permissive, but it is the reference behaviour.
    transparency = parse(
        (
            "#......",
            "#......",
            ".......",
            "#......",
            "#......",
            "######.",
            "#......",
        ),
        ".",
    )
    assert bool(jnp.all(process_vis(transparency)))


def test_process_vis_hides_what_a_solid_wall_blocks():
    # The same window with the gap closed: now nothing past the wall is
    # reachable, and only the wall itself and the agent's row remain.
    transparency = parse(
        (
            "#......",
            "#......",
            ".......",
            "#......",
            "#......",
            "#######",
            "#......",
        ),
        ".",
    )
    seen = process_vis(transparency)
    assert not bool(jnp.any(seen[:5])), np.asarray(seen)
    assert bool(jnp.all(seen[5:])), np.asarray(seen)


def test_process_vis_is_all_visible_in_an_empty_room():
    seen = process_vis(jnp.ones((7, 7), dtype=jnp.bool))
    assert bool(jnp.all(seen))


def test_process_vis_sees_only_the_agent_when_boxed_in():
    transparency = jnp.ones((7, 7), dtype=jnp.bool).at[5, :].set(False)
    seen = process_vis(transparency)
    # The blocking row is still seen; nothing past it is.
    assert not bool(jnp.any(seen[:5]))
    assert bool(jnp.all(seen[5:]))


def test_process_vis_returns_bool():
    seen = process_vis(jnp.ones((7, 7), dtype=jnp.int32))
    assert seen.dtype == jnp.bool


def test_process_vis_is_jittable_and_batchable():
    grids = jnp.asarray(np.random.default_rng(0).random((16, 7, 7)) > 0.3)
    batched = jax.jit(jax.vmap(process_vis))(grids)
    assert batched.shape == grids.shape
    for i in range(grids.shape[0]):
        np.testing.assert_array_equal(
            np.asarray(batched[i]), np.asarray(process_vis(grids[i]))
        )


def test_process_vis_rejects_a_window_with_no_centre():
    # The agent's place in the window is a convention shared with crop(),
    # not something the window states, so an even width means the crop
    # layout changed underneath it rather than a window it can answer for.
    with pytest.raises(ValueError, match="bottom-centre"):
        process_vis(jnp.ones((7, 8), dtype=jnp.bool))


def test_process_vis_does_not_wrap_around_the_row():
    # Standing in the corner of an all-transparent grid facing east: the
    # crop pads the two columns that fall off the map with opaque cells,
    # and sight must stop at the first of them rather than wrapping
    # around the row edge the way a roll-based flood would.
    transparency = jnp.ones((9, 9), dtype=jnp.bool)
    window = crop(transparency, jnp.asarray((1, 1)), jnp.asarray(0), 3, padding_value=0)
    seen = process_vis(window > 0)
    assert bool(jnp.all(seen[:, 1])), "the padding cell itself should be seen"
    assert not bool(jnp.any(seen[:, 0])), "sight wrapped around the row"


if __name__ == "__main__":
    test_process_vis_matches_minigrid()
    test_process_vis_flows_through_a_gap_in_a_wall()
    test_process_vis_hides_what_a_solid_wall_blocks()
    test_process_vis_is_all_visible_in_an_empty_room()
    test_process_vis_sees_only_the_agent_when_boxed_in()
    test_process_vis_returns_bool()
    test_process_vis_is_jittable_and_batchable()
    test_process_vis_rejects_a_window_with_no_centre()
    test_process_vis_does_not_wrap_around_the_row()
