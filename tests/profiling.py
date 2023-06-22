import jax
import jax.numpy as jnp
import navix as nx


N_TIMESTEPS = 100


with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    seed = 0
    env = nx.environments.Room(16, 16, 8, observation_fn=nx.observations.rgb)
    key = jax.random.PRNGKey(seed)
    timestep = env.reset(key)
    actions = jax.random.randint(key, (N_TIMESTEPS,), 0, 6)

    # def body_fun(carry, x):
    #     timestep = carry
    #     action = x
    #     timestep = env.step(timestep, jnp.asarray(action))
    #     return timestep, ()
    # timestep = jax.lax.scan(body_fun, timestep, jnp.asarray(actions, dtype=jnp.int32))[0]

    for i in range(N_TIMESTEPS):
        timestep = env.step(timestep, actions[i])

    timestep.observation.block_until_ready()
