from __future__ import annotations

import os
import jax
import jax.numpy as jnp

import navix as nx
from navix import observations

import matplotlib.pyplot as plt


def test_82():
    os.environ["JAX_OMNISTAGING"] = "0"
    def render(timestep):
        plt.imshow(timestep.observation)
        plt.axis(False)
        plt.savefig("test_82.png")

    env = nx.make(
        "Navix-DoorKey-5x5-v0",
        max_steps=100,
        observation_fn=observations.rgb,
    )
    key = jax.random.PRNGKey(5)
    timestep = env.reset(key)
    # Seed 5 is:
    # # # # #
    # P # . #
    # . # . #
    # K D G #
    # # # # #
    
    # start agent direction = EAST
    prev_pos = timestep.state.entities["player"].position
    # action 2 is forward
    timestep = env.step(timestep, 2)  # should not walk into wall
    pos = timestep.state.entities["player"].position
    assert jnp.array_equal(prev_pos, pos)


if __name__ == "__main__":
    test_82()