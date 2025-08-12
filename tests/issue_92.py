"""Unittest for issue #92: https://github.com/epignatelli/navix/issues/92"""

import jax
import navix as nx
from navix.states import Event, EventType


def test_issue_92():
    # try instantiate Event
    event = Event(
        position=jax.numpy.asarray([-1, -1]),
        colour=jax.numpy.asarray([255, 0, 0]),
        happened=jax.numpy.asarray(False),
        event_type=EventType.NONE,
    )
    # try instantiate environment
    env = nx.make("Navix-KeyCorridorS6R3-v0")
    timestep = env.reset(jax.random.PRNGKey(0))
    timestep = env.step(timestep, jax.numpy.asarray(1))
