import jax
import jax.numpy as jnp
import navix as nx
from navix.grid import view_cone


def test_grid_from_ascii():
    ascii_map = """########
    #1.....#
    #......#
        #......#
#......#
        #......#
    #......#
    #.....2#
########
########
########
########
    """
    print(ascii_map)

    grid = nx.grid.from_ascii_map(ascii_map)
    print(grid)

    ascii_map = ascii_map.replace("1", "P")
    ascii_map = ascii_map.replace("2", "G")
    grid = nx.grid.from_ascii_map(ascii_map, mapping={"P": 1, "G": 2})
    print(grid)


def test_idx_from_coordinates():
    grid = jnp.zeros((5, 7), dtype=jnp.int32)
    grid = jnp.pad(grid, 1, mode="constant", constant_values=-1)

    positions = jnp.array([[1, 1], [2, 2], [3, 3], [4, 4]])
    indices = nx.grid.idx_from_coordinates(grid, positions)
    positions_after = nx.grid.coordinates_from_idx(grid, indices)
    assert jnp.all(jnp.array_equal(positions, positions_after)), (
        positions,
        positions_after,
    )


def test_random_positions():
    grid = jnp.zeros((5, 7), dtype=jnp.int32)
    grid = jnp.pad(grid, 1, mode="constant", constant_values=-1)

    key = jax.random.PRNGKey(0)
    positions = nx.grid.random_positions(key, grid, n=1)
    assert positions.shape == (2,), positions.shape

    positions = nx.grid.random_positions(key, grid, n=4)
    assert positions.shape == (4, 2), positions.shape

    exclude = jnp.asarray((1, 1))
    positions = nx.grid.random_positions(key, grid, n=50, exclude=exclude)
    for position in positions:
        assert not jnp.array_equal(position, exclude), position
        assert jnp.array_equal(grid[tuple(position)], 0), positions


def test_position_equal():
    # one to one
    a = jnp.array([1, 1])
    b = jnp.array([1, 1])
    assert nx.grid.positions_equal(a, b)
    assert nx.grid.positions_equal(b, a)
    assert not nx.grid.positions_equal(a, b + 1)
    assert not nx.grid.positions_equal(a + 1, b)
    assert not nx.grid.positions_equal(b, a + 1)
    assert not nx.grid.positions_equal(b + 1, a)

    # one to many
    a = jnp.array([1, 1])
    b = jnp.array([[1, 1], [1, 2]])
    assert jnp.array_equal(nx.grid.positions_equal(a, b), jnp.array([True, False]))
    assert jnp.array_equal(nx.grid.positions_equal(b, a), jnp.array([True, False]))
    assert jnp.array_equal(nx.grid.positions_equal(a, b + 1), jnp.array([False, False]))
    assert jnp.array_equal(nx.grid.positions_equal(a + 1, b), jnp.array([False, False]))
    assert jnp.array_equal(nx.grid.positions_equal(b, a + 1), jnp.array([False, False]))
    assert jnp.array_equal(nx.grid.positions_equal(b + 1, a), jnp.array([False, False]))

    # many to many
    a = jnp.array([[1, 1], [1, 2]])
    b = jnp.array([[1, 1], [1, 2]])
    assert jnp.array_equal(nx.grid.positions_equal(a, b), jnp.array([True, True]))
    assert jnp.array_equal(nx.grid.positions_equal(b, a), jnp.array([True, True]))
    assert jnp.array_equal(nx.grid.positions_equal(a, b + 1), jnp.array([False, False]))
    assert jnp.array_equal(nx.grid.positions_equal(a + 1, b), jnp.array([False, False]))
    assert jnp.array_equal(nx.grid.positions_equal(b, a + 1), jnp.array([False, False]))
    assert jnp.array_equal(nx.grid.positions_equal(b + 1, a), jnp.array([False, False]))


def test_view_cone_does_not_overflow_at_large_radius():
    # view_cone's fin_diff accumulates path *counts* (each step lets
    # every unit of "mass" split into up to 9 copies of itself, a 3x3
    # neighbourhood sum), not just reachability - the total grows
    # ~9x per step and only `view > 0` is ever read downstream. In an
    # open area this overflows int32 (~2.1e9) at cone radius 12; past
    # that, a wrapped-negative cell is indistinguishable from a
    # genuinely unreachable one, so overflow silently *removes*
    # visibility instead of erroring. Found and verified independently
    # by @Near32 on PR #148, who proposed clamping to a boolean flood
    # (min with 1) after every step, since the magnitude is discarded
    # by the `> 0` threshold anyway.
    #
    # For an open room with the origin far enough from every wall that
    # the whole diffusion stays in-bounds, a `radius`-step king-move
    # flood should reach exactly a (2*radius+1) x (2*radius+1) square -
    # verified against that closed form, rather than a fixed magic
    # number, so the assertion holds at any radius.
    for radius in [10, 11, 12, 13, 14, 24]:  # straddles the old radius=12 overflow
        n = 4 * radius + 9  # room comfortably larger than the flood's reach
        transparency_map = jnp.ones((n, n), dtype=jnp.int32)
        transparency_map = (
            transparency_map.at[0, :].set(0)
            .at[-1, :].set(0)
            .at[:, 0].set(0)
            .at[:, -1].set(0)
        )
        origin = jnp.asarray([n // 2, n // 2])
        view = view_cone(transparency_map, origin, radius)
        expected = (2 * radius + 1) ** 2
        assert int(view.sum()) == expected, (
            f"radius={radius}: expected a full {2 * radius + 1}x{2 * radius + 1} "
            f"visible square ({expected} cells), got {int(view.sum())} - some "
            "cells were likely dropped by int32 overflow in the accumulator"
        )


if __name__ == "__main__":
    # test_grid_from_ascii()
    # test_idx_from_coordinates()
    # test_random_positions()
    test_position_equal()
    # jax.jit(test_position_equal)()
    test_view_cone_does_not_overflow_at_large_radius()
