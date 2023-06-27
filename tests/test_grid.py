import jax
import jax.numpy as jnp
import navix as nx


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


if __name__ == "__main__":
    # test_grid_from_ascii()
    # test_idx_from_coordinates()
    # test_random_positions()
    test_position_equal()
    # jax.jit(test_position_equal)()
