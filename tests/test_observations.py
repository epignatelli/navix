import jax
import jax.numpy as jnp

import navix as nx


def test_rgb():
    height = 10
    width = 10
    grid = jnp.zeros((height - 2, width - 2), dtype=jnp.int32)
    grid = jnp.pad(grid, 1, mode="constant", constant_values=-1)

    state = nx.components.State(
        key=jax.random.PRNGKey(0),
        grid=grid,
        player=nx.components.Player(position=jnp.asarray((1, 1))),
        goals=nx.components.Goal(position=jnp.asarray((4, 4))[None]),
        keys=nx.components.Pickable(position=jnp.asarray((2, 2))[None]),
        doors=nx.components.Consumable(position=jnp.asarray((1, 5))[None]),
        cache=nx.graphics.RenderingCache.init(grid),
    )
    void = jnp.zeros((nx.graphics.TILE_SIZE, nx.graphics.TILE_SIZE, 3), dtype=jnp.uint8)
    tiles_registry={
        "player": void,
        "goal": void,
        "key": void,
        "door": void,
    }

    obs = nx.observations.rgb(state, tiles_registry=tiles_registry)
    expected_obs_shape = (height * nx.graphics.TILE_SIZE, width * nx.graphics.TILE_SIZE, 3)
    assert obs.shape == expected_obs_shape, (
        f"Expected observation {expected_obs_shape}, got {obs.shape} instead"
    )

    def get_tile(position):
        x = position[0] * nx.graphics.TILE_SIZE
        y = position[1] * nx.graphics.TILE_SIZE
        return obs[x:x + nx.graphics.TILE_SIZE, y:y + nx.graphics.TILE_SIZE, :]

    player_tile = get_tile(state.player.position)
    assert jnp.array_equal(player_tile, tiles_registry["player"]), player_tile

    goal_tile = get_tile(state.goals.position[0])
    assert jnp.array_equal(goal_tile, tiles_registry["goal"]), goal_tile

    key_tile = get_tile(state.keys.position[0])
    assert jnp.array_equal(key_tile, tiles_registry["key"]), key_tile

    door_tile = get_tile(state.doors.position[0])
    assert jnp.array_equal(door_tile, tiles_registry["door"]), door_tile

    return


if __name__ == "__main__":
    test_rgb()
