from __future__ import annotations

import os
from typing import Dict, Tuple
from PIL import Image

import jax
from jax import Array
import jax.numpy as jnp


SPRITES_DIR = os.path.normpath(
    os.path.join(__file__, "..", "..", "..", "assets", "sprites")
)
MIN_TILE__SIZE = 8
TILE_SIZE = 32


def load_sprite(name: str) -> Array:
    """Loads an image from disk in RGB space.
    Args:
        path(str): the filepath of the image on disk

    Returns:
        (Array): a jax.Array of shape (H, W, C)"""
    path = os.path.join(SPRITES_DIR, f"{name}.png")
    image = Image.open(path)
    array = jnp.asarray(image)
    resized = jax.image.resize(array, (TILE_SIZE, TILE_SIZE, 3), method="nearest")
    return resized


class Colours:
    RED: str = "red"
    GREEN: str = "green"
    BLUE: str = "blue"
    PURPLE: str = "purple"
    YELLOW: str = "yellow"
    GREY: str = "grey"


COLOURS = [
    Colours.RED,
    Colours.GREEN,
    Colours.BLUE,
    Colours.PURPLE,
    Colours.YELLOW,
    Colours.GREY,
]


class SpritesRegistry:
    def __init__(self):
        self._registry = {}
        self.build_registry()

    def __getitem__(self, key: str | Tuple[str, str]) -> Array:
        return self.registry[key]

    @property
    def registry(self) -> Dict[str | Tuple[str, str], Array]:
        # lazy init
        if self._registry is None:
            self.build_registry()
        return self._registry

    def get(
        self,
        name: str,
        direction: Array | None = None,
        closed: Array | None = None,
        colour: str | None = None,
    ) -> Array:
        key = name
        if colour is not None:
            key = (name, colour)
        sprite = self._registry[key]
        if direction is not None:
            sprite = sprite[direction]
        if closed is not None:
            sprite = sprite[closed]
        return sprite

    def build_registry(self):
        """Populates the sprites registry for all entities."""
        self.set_wall_sprite()
        self.set_floor_sprite()
        self.set_goal_sprite()
        self.set_key_sprite()
        self.set_player_sprite()
        self.set_door_sprite()

    def set_wall_sprite(self):
        self._registry["wall"] = load_sprite("wall")

    def set_floor_sprite(self):
        self._registry["floor"] = load_sprite("floor")

    def set_goal_sprite(self):
        self._registry["goal"] = load_sprite("goal")

    def set_key_sprite(self):
        for colour in COLOURS:
            self._registry["key", colour] = load_sprite("key" + f"_{colour}")

    def set_player_sprite(self):
        self._registry["player"] = jnp.stack(
            [
                load_sprite("player_east"),
                load_sprite("player_south"),
                load_sprite("player_west"),
                load_sprite("player_north"),
            ]
        )

    def set_door_sprite(self):
        for colour in COLOURS:
            door_closed_sprite = load_sprite("door" + "_closed" + f"_{colour}")
            door_open_sprite = load_sprite("door" + "_open" + f"_{colour}")
            door_locked_sprite = load_sprite("door" + "_locked" + f"_{colour}")

            # cannot use tuples here, see https://github.com/google/jax/issues/16559
            door = jnp.zeros((3, TILE_SIZE, TILE_SIZE, 3), dtype=jnp.uint8)

            door = door.at[0].set(door_closed_sprite)
            door = door.at[1].set(door_open_sprite)
            door = door.at[2].set(door_locked_sprite)

            self._registry["door", colour] = door


# initialise sprites registry
SPRITES_REGISTRY = SpritesRegistry()
