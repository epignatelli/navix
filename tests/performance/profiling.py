import jax
import jax.numpy as jnp
import navix as nx


N_TIMESTEPS = 1000


def f():
    seed = 0
    env = nx.environments.Room(16, 16, 8, observation_fn=nx.observations.rgb_first_person)
    key = jax.random.PRNGKey(seed)
    timestep = env.reset(key)
    timestep = env.step(timestep, jnp.asarray(3))
    timestep = env.step(timestep, jnp.asarray(3))
    timestep = env.step(timestep, jnp.asarray(3))
    timestep = env.step(timestep, jnp.asarray(3))
    timestep = env.step(timestep, jnp.asarray(3))
    timestep = env.step(timestep, jnp.asarray(3))
    timestep = env.step(timestep, jnp.asarray(3))
    return timestep


def f_scan():
    seed = 0
    env = nx.environments.Room(16, 16, 8, observation_fn=nx.observations.rgb)
    key = jax.random.PRNGKey(seed)
    timestep = env.reset(key)

    def body_fun(carry, x):
        timestep = carry
        action = x
        timestep = env.step(timestep, jnp.asarray(action))
        return timestep, ()

    actions = jax.random.randint(key, (N_TIMESTEPS,), 0, 6)
    timestep = jax.lax.scan(
        lambda c, x: (env.step(c, x), ()),
        timestep,
        jnp.asarray(actions, dtype=jnp.int32),
    )[0]
    return timestep


f_jit = jax.jit(f)
f_scan_jit = jax.jit(f_scan)

# warm up
f_jit()
f_scan_jit()

with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    timestep = f_scan()
    timestep.observation.block_until_ready()
