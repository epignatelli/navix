import jax
import jax.numpy as jnp
import navix as nx


def test_grid():
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


def test_random_positions():
    def f():
        env = nx.environments.KeyDoor(18, 6, 100)
        key = jax.random.PRNGKey(7)
        timestep = env.reset(key)
        # without the `exclude` params in `random_positions` this
        # specific configuration of seed and (width, height) draws player
        # pos [2, 1] and key pos [2, 1] check that this does not happen anymore
        return timestep.state.player.position, timestep.state.keys.position[0]

    player_pos, key_pos = f()
    assert not jnp.array_equal(player_pos, key_pos)
    player_pos, key_pos = jax.jit(f)()
    assert not jnp.array_equal(player_pos, key_pos)


if __name__ == "__main__":
    # test_grid()
    test_random_positions()
