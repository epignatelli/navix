import jax
import jax.numpy as jnp
import navix as nx
import time


N_TIMESTEPS = 10
N_SEEDS = 100


def f(seed):
    key = jax.random.PRNGKey(seed)
    env = nx.environments.Room(16, 16, 8, observation_fn=nx.observations.rgb)
    timestep = env._reset(key)

    for _ in range(N_TIMESTEPS):
        action = jax.random.randint(timestep.state.key, (), 0, 6)
        timestep = env.step(timestep, jnp.asarray(action))
    return timestep


def f_scan(seed):
    key = jax.random.PRNGKey(seed)
    env = nx.environments.Room(16, 16, 8, observation_fn=nx.observations.rgb)
    timestep = env._reset(key)

    def body_fun(carry, x):
        timestep = carry
        action = jax.random.randint(timestep.state.key, (), 0, 6)
        timestep = env.step(timestep, action)
        return timestep, ()

    timestep = jax.lax.scan(
        body_fun,
        timestep,
        [None] * N_TIMESTEPS,
        length=N_TIMESTEPS,
    )[0]
    return timestep


seeds = jnp.arange(N_SEEDS)
function = jax.vmap(f)

# print(f"\tCompiling {function}...")
# start = time.time()
# f_jit = jax.jit(function).lower(seeds).compile()
# print("\tCompiled in {:.2f}s".format(time.time() - start))


with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    timestep = function(seeds).observation.block_until_ready()
