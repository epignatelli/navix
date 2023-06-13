from __future__ import annotations

from chex import Array

from .components import State


def deterministic_transition(state: State, action: Array) -> State:
    return state

def windy_transition(state: State, action: Array) -> State:
    raise NotImplementedError()