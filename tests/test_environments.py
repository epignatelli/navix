import jax
import jax.numpy as jnp
import navix as nx


def test_room():
    def f():
        env = nx.environments.Room.create(
            height=3,
            width=3,
            max_steps=8,
            observation_fn=nx.observations.symbolic_first_person,
        )
        key = jax.random.PRNGKey(4)
        reset = jax.jit(env._reset)
        step = jax.jit(env.step)
        timestep = reset(key)
        # these are optimal actios for navigation + action_cost
        actions = (
            0,  # noop sanity check
            2,  # rotate_ccw
            3,  # forward
            3,  # forward
            2,  # rotate_ccw
            3,  # forward
        )
        print(timestep)
        print()
        for action in actions:
            timestep = step(timestep, jnp.asarray(action))
            print()
            print(nx.actions.DEFAULT_ACTION_SET[action])
            print(timestep)
        return timestep

    f()
    timestep = jax.jit(f)()
    print(timestep)


def test_keydoor():
    def f():
        env = nx.environments.DoorKey.create(
            height=5,
            width=10,
            max_steps=8,
            observation_fn=nx.observations.symbolic_first_person,
        )
        key = jax.random.PRNGKey(1)
        reset = jax.jit(env._reset)
        step = jax.jit(env.step)
        timestep = reset(key)
        #  these are optimal actions for navigation + action_cost
        actions = (
            0,  # rotate_ccw
            2,  # forward
            2,  # forward
            2,  # forward
            0,  # rotate_ccw
            3,  # pick-up
            0,  # rotate_ccw
            0,  # rotate_ccw
            2,  # forward
            2,  # forward
            1,  # rotate_cw
            2,  # forward
            0,  # rotate_ccw
            5,  # open
            2,  # forward
            2,  # forward
        )
        print(timestep)
        for action in actions:
            timestep = step(timestep, jnp.asarray(action))
            print()
            print(nx.actions.DEFAULT_ACTION_SET[action])
            print(timestep)
        return timestep

    f()
    jax.jit(f)()


def test_keydoor2():
    env = nx.environments.DoorKey.create(5, 7, 100, observation_fn=nx.observations.rgb)

    key = jax.random.PRNGKey(1)
    timestep = env._reset(key)
    return


def test_fourrooms_sizes_partition_walls_are_centred():
    # the partition walls used to be hardcoded at row/col 9, which only
    # matched the original 19x19 default; for the smaller registered sizes
    # (7x7 .. 17x17) that put the wall out of bounds or off-centre, so the
    # room was never actually split into four rooms
    for height, width in ((7, 7), (9, 9), (11, 11), (13, 13), (15, 15), (17, 17)):
        env = nx.environments.FourRooms.create(
            height=height, width=width, observation_fn=nx.observations.none
        )
        timestep = env.reset(jax.random.PRNGKey(0))
        positions = timestep.state.get_walls().position
        rows, cols = positions[:, 0], positions[:, 1]

        on_cross = (rows == height // 2) | (cols == width // 2)
        assert jnp.all(on_cross), (
            f"FourRooms {height}x{width}: partition walls {positions} are not "
            f"centred at row={height // 2} / col={width // 2}"
        )
        assert jnp.all((rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)), (
            f"FourRooms {height}x{width}: partition walls {positions} fall "
            "outside the grid"
        )


def test_disable_autoreset():
    def make_env(disable):
        return nx.environments.Room.create(
            height=5,
            width=5,
            max_steps=2,
            observation_fn=nx.observations.none,
            disable_autoreset=disable,
        )

    key = jax.random.PRNGKey(0)

    env = make_env(disable=True)
    timestep = env.reset(key)
    for _ in range(5):
        timestep = env.step(timestep, jnp.asarray(0))
    assert (
        timestep.step_type > 0
    ), "expected step_type to stay terminal/truncated when disable_autoreset=True"
    assert (
        timestep.t > env.max_steps
    ), "expected t to keep advancing past max_steps instead of autoresetting"

    env2 = make_env(disable=False)
    timestep2 = env2.reset(key)
    for _ in range(5):
        timestep2 = env2.step(timestep2, jnp.asarray(0))
    assert (
        timestep2.t <= env2.max_steps
    ), "expected autoreset to reset t back down by default"


if __name__ == "__main__":
    # test_room()
    # jax.jit(test_room)()
    test_keydoor()
    # test_keydoor2()
