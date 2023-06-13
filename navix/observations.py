from __future__ import annotations

from chex import Array

from .components import State


def third_person_view(state: State, radius: int) -> Array:
    raise NotImplementedError()

def first_person_view(state: State, radius: int) -> Array:
    raise NotImplementedError()

def categorical(state: State) -> Array:
    raise NotImplementedError()

def one_hot(state: State) -> Array:
    raise NotImplementedError()

def pixels(state: State) -> Array:
    raise NotImplementedError()