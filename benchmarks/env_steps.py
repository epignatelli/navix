from dataclasses import dataclass
import os
import time
import json
from typing import Tuple

import gymnasium as gym
from gymnasium.wrappers.autoreset import AutoResetWrapper
import jax
import tyro
import navix as nx
from navix.environments.environment import Environment, Timestep


NAVIX_TO_MINIGRID = {
    "Navix-Empty-5x5-v0": "MiniGrid-Empty-5x5-v0",
    "Navix-Empty-6x6-v0": "MiniGrid-Empty-6x6-v0",
    "Navix-Empty-8x8-v0": "MiniGrid-Empty-8x8-v0",
    "Navix-Empty-16x16-v0": "MiniGrid-Empty-16x16-v0",
    "Navix-DoorKey-5x5-v0": "MiniGrid-DoorKey-5x5-v0",
    "Navix-DoorKey-6x6-v0": "MiniGrid-DoorKey-6x6-v0",
    "Navix-DoorKey-8x8-v0": "MiniGrid-DoorKey-8x8-v0",
    "Navix-DoorKey-16x16-v0": "MiniGrid-DoorKey-16x16-v0",
    "Navix-FourRooms-v0": "MiniGrid-FourRooms-v0",
    "Navix-KeyCorridorS3R1-v0": "MiniGrid-KeyCorridorS3R1-v0",
    "Navix-KeyCorridorS3R2-v0": "MiniGrid-KeyCorridorS3R2-v0",
    "Navix-KeyCorridorS3R3-v0": "MiniGrid-KeyCorridorS3R3-v0",
    "Navix-KeyCorridorS4R3-v0": "MiniGrid-KeyCorridorS4R3-v0",
    "Navix-KeyCorridorS5R3-v0": "MiniGrid-KeyCorridorS5R3-v0",
    "Navix-KeyCorridorS6R3-v0": "MiniGrid-KeyCorridorS6R3-v0",
    "Navix-LavaGap-S5-v0": "MiniGrid-LavaGapS5-v0",
    "Navix-LavaGap-S6-v0": "MiniGrid-LavaGapS6-v0",
    "Navix-LavaGap-S7-v0": "MiniGrid-LavaGapS7-v0",
    "Navix-Crossings-S9N1-v0": "MiniGrid-SimpleCrossingS9N1-v0",
    "Navix-Crossings-S9N2-v0": "MiniGrid-SimpleCrossingS9N2-v0",
    "Navix-Crossings-S9N3-v0": "MiniGrid-SimpleCrossingS9N3-v0",
    "Navix-Crossings-S11N5-v0": "MiniGrid-SimpleCrossingS11N5-v0",
    "Navix-Dynamic-Obstacles-5x5": "MiniGrid-Dynamic-Obstacles-5x5-v0",
    "Navix-Dynamic-Obstacles-6x6": "MiniGrid-Dynamic-Obstacles-6x6-v0",
    "Navix-Dynamic-Obstacles-8x8": "MiniGrid-Dynamic-Obstacles-8x8-v0",
    "Navix-Dynamic-Obstacles-16x16": "MiniGrid-Dynamic-Obstacles-16x16-v0",
    "Navix-DistShift1-v0": "MiniGrid-DistShift1-v0",
    "Navix-DistShift2-v0": "MiniGrid-DistShift2-v0",
    "Navix-GoToDoor-5x5-v0": "MiniGrid-GoToDoor-5x5-v0",
    "Navix-GoToDoor-6x6-v0": "MiniGrid-GoToDoor-6x6-v0",
    "Navix-GoToDoor-8x8-v0": "MiniGrid-GoToDoor-8x8-v0",
}


@dataclass
class Args:
    budget: int = 1_000_000
    num_runs: int = 5


def collect_experience(env: Environment, timestep: Timestep, length: int) -> Timestep:
    key = jax.random.PRNGKey(0)
    # actions = jax.random.randint(key, (length,), 0, env.action_space.maximum + 1)
    action = jax.random.randint(key, (), 0, env.action_space.maximum + 1)

    def _env_step(env_state: Timestep, _) -> Tuple[Timestep, tuple]:
        # STEP ENV
        # new_env_state = jax.vmap(env.step, in_axes=(0, 0))(env_state, action)
        new_env_state = env.step(env_state, action)
        return new_env_state, ()

    # collect experience and update env_state
    timestep, _ = jax.lax.scan(
        _env_step,
        timestep,
        None,
        length,
        unroll=100,
    )
    return timestep


def run_navix(env_id: str, length: int) -> float:
    print("Initialising navix env...")
    env = nx.make(env_id)
    key = jax.random.PRNGKey(0)
    timestep = env.reset(key)

    print("Compiling navix function")
    compile_start = time.perf_counter()
    unroll_jit = (
        jax.jit(collect_experience, static_argnums=(2,))
        .lower(env, timestep, length)
        .compile()
    )
    compile_end = time.perf_counter()
    print("Compiled navix function in", compile_end - compile_start)

    print("Running navix")
    start_time = time.perf_counter()
    timestep = unroll_jit(env, timestep)
    timestep.t.block_until_ready()
    end_time = time.perf_counter()
    print("Completed navix", env_id, end_time - start_time)

    return end_time - start_time


def run_minigrid(env_id: str, length: int) -> float:
    print("Initialising minigrid env...")
    env = AutoResetWrapper(gym.make(NAVIX_TO_MINIGRID[env_id]))
    env.reset()

    actions = jax.random.randint(jax.random.PRNGKey(0), (length,), 0, 7)

    def unroll_minigrid():
        for action in actions:
            timestep = env.step(action)
        return timestep

    print("Running minigrid")
    start_time = time.perf_counter()
    unroll_minigrid()
    end_time = time.perf_counter()
    print("Completed minigrid", env_id, end_time - start_time)

    return end_time - start_time


def main():
    out_path = os.path.join(os.path.dirname(__file__), "env_steps.json")
    env_id = "Navix-Empty-5x5-v0"

    results = {}
    for i in range(args.num_runs):
        print("Run navix", i)
        elapsed = run_navix(env_id, args.budget)
        # results[env_id] = results.get(env_id, []) + [elapsed]

    for i in range(args.num_runs):
        elapsed = run_minigrid(env_id, args.budget)
        minigrid_id = NAVIX_TO_MINIGRID[env_id]
        # results[minigrid_id] = results.get(minigrid_id, []) + [elapsed]

    print("Writing results")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    args = tyro.cli(Args)

    main()
