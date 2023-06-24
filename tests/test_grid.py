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

    positions = nx.grid.random_positions(key, grid, n=4, exclude=jnp.asarray((1, 1)))
    assert jnp.all(positions != jnp.asarray((1, 1))), positions


if __name__ == "__main__":
    test_grid_from_ascii()
    test_idx_from_coordinates()
    test_random_positions()
