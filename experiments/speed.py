import os
import timeit
import json
import gymnasium as gym
import jax
import navix as nx


def run_minigrid(env_id: str, num_steps: int, num_runs: int):
    print("Running MiniGrid...")

    def _run():
        env = gym.make(env_id, max_episode_steps=num_steps)
        env.reset()

        for _ in range(num_steps):
            action = env.action_space.sample()
            timestep = env.step(action)
        return timestep

    times = timeit.timeit(_run, number=num_runs)
    return times


def run_navix_jit_step(env_id: str, num_steps: int, num_runs: int):
    print("Running Navix JIT step...")
    env = nx.make(env_id, max_steps=num_steps)

    key = jax.random.PRNGKey(0)
    timestep = env.reset(key)
    step = jax.jit(env.step).lower(timestep, env.action_space.sample(key)).compile()

    def _run(timestep):
        for i in range(num_steps):
            action = env.action_space.sample(key)
            timestep = step(timestep, action)
        return timestep

    times = timeit.timeit(lambda: _run(timestep).t.block_until_ready(), number=num_runs)
    return times


def run_navix_jit_loop(env_id: str, num_steps: int, num_runs: int):
    print("Running Navix JIT loop...")

    def _run(key):
        env = nx.make(env_id, max_steps=num_steps)  # Create the environment
        timestep = env.reset(key)
        actions = jax.random.randint(key, (num_steps,), 0, env.action_space.n)

        def body_fun(timestep, action):
            timestep = env.step(timestep, action)  # Update the environment state
            return timestep, ()

        return jax.lax.scan(body_fun, timestep, actions, unroll=10)[0]

    key = jax.random.PRNGKey(0)
    run = jax.jit(_run).lower(key).compile()
    times = timeit.timeit(lambda: run(key).t.block_until_ready(), number=num_runs)
    return times


def speedup_by_env():
    print("*" * 80)
    print("Running speedup by env...")
    print("*" * 80)
    # NUM_STEPS = 1_000
    NUM_STEPS = 10
    NUM_RUNS = 1

    results = {}
    for env_id in nx.registry():
        try:
            gym_env_id = env_id.replace("Navix", "MiniGrid")
            print(env_id)
            results[env_id] = {
                "minigrid": run_minigrid(gym_env_id, NUM_STEPS, NUM_RUNS),
                "navix_jit_step": run_navix_jit_step(env_id, NUM_STEPS, NUM_RUNS),
                "navix_jit_loop": run_navix_jit_loop(env_id, NUM_STEPS, NUM_RUNS),
            }
            with open(
                os.path.join(os.path.dirname(__file__), "speedup_env.json"), "w"
            ) as f:
                json.dump(results, f, indent=2)
        except KeyboardInterrupt:
            return
        except Exception as e:
            print(f"Error in {env_id}: {repr(e)}")


def speedup_by_num_steps():
    print("*" * 80)
    print("Running speedup by num steps")
    print("*" * 80)
    NUM_RUNS = 1
    ENV_ID = "Navix-Empty-8x8-v0"
    gym_env_id = ENV_ID.replace("Navix", "MiniGrid")

    results = {}
    for order in range(1, 7):
        num_steps = 10**order
        print(num_steps)
        results[num_steps] = {
            "minigrid": run_minigrid(gym_env_id, num_steps, NUM_RUNS),
            "navix_jit_step": run_navix_jit_step(ENV_ID, num_steps, NUM_RUNS),
            "navix_jit_loop": run_navix_jit_loop(ENV_ID, num_steps, NUM_RUNS),
        }
        with open(
            os.path.join(os.path.dirname(__file__), "speedup_num_steps.json"), "w"
        ) as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    speedup_by_num_steps()
    speedup_by_env()
